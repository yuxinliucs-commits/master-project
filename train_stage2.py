"""Stage 2: identity-encoding training with multi-positive contrastive loss and 3D KL
regularization (paper Sec. 3.5.3 / 3.5.4 / 4.1: 10K iterations, geometry frozen)."""
import os
from random import randint
from argparse import ArgumentParser

import torch
from tqdm import tqdm

from mcggs.config import load_config
from mcggs.scene import Scene
from mcggs.scene.gaussian_model import GroupedGaussianModel
from mcggs.gaussian_renderer import render
from mcggs.losses.contrastive import multi_positive_contrastive_loss
from mcggs.losses.regularization import kl_regularization_3d
from mcggs.utils.loss_utils import l1_loss, ssim
from mcggs.utils.image_utils import psnr
from mcggs.utils.general_utils import set_seed


class IdentityHead(torch.nn.Module):
    """Linear layer 16 -> 256 applied to rendered identity features / raw encodings (paper 4.1)."""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = torch.nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.linear(x)


def training(cfg):
    data = cfg["data"]
    s1 = cfg["stage1"]
    s2 = cfg["stage2"]

    gaussians = GroupedGaussianModel(data["sh_degree"], identity_dim=s2["identity_dim"])
    scene = Scene(cfg, gaussians)
    gaussians.freeze_geometry()

    head = IdentityHead(s2["identity_dim"], s2["embed_dim"]).cuda()
    gaussians.training_setup(cfg, stage="stage2", embed_head=head)

    bg_color = [1, 1, 1] if data["white_background"] else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    pipe = cfg["pipeline"]
    masks_dir = os.path.join(data["source_path"], data["masks_dir"])
    assert os.path.isdir(masks_dir), \
        "associated masks not found at %s; run associate_masks.py first" % masks_dir

    train_cams = scene.getTrainCameras()
    # preload associated label maps once (paper 3.5.1: pseudo-labels are fixed before stage 2)
    label_cache = {}
    for cam_k in train_cams:
        p = os.path.join(masks_dir, cam_k.image_name + ".png")
        if os.path.exists(p):
            import numpy as np
            from PIL import Image
            m = np.array(Image.open(p)).astype(np.int64)
            if m.shape != (cam_k.image_height, cam_k.image_width):
                m = np.array(Image.fromarray(m.astype(np.int32)).resize(
                    (cam_k.image_width, cam_k.image_height), Image.NEAREST))
            label_cache[cam_k.image_name] = torch.from_numpy(m.astype(np.int64)).cuda()
        else:
            label_cache[cam_k.image_name] = None
    n_labeled = sum(1 for v in label_cache.values() if v is not None)
    print("[stage2] loaded associated labels for %d/%d views" % (n_labeled, len(train_cams)))

    viewpoint_stack = []
    ema = 0.0
    progress = tqdm(range(1, s2["iterations"] + 1), desc="stage2")
    for iteration in progress:
        # ---- reconstruction loss on one random view (grad reaches only identity & head) ----
        if not viewpoint_stack:
            viewpoint_stack = train_cams.copy()
        cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        render_pkg = render(cam, gaussians, pipe, background, use_identity=True)
        image = render_pkg["render"]
        gt_image = cam.original_image.cuda()
        L_recon = (1.0 - s1["lambda_dssim"]) * l1_loss(image, gt_image) + \
            s1["lambda_dssim"] * (1.0 - ssim(image, gt_image))

        # ---- multi-positive contrastive loss over sampled views (paper 3.5.3) ----
        n_views = s2.get("views_per_step", 2)
        id_maps, lab_maps = [], []
        for k in range(n_views):
            cam_k = train_cams[randint(0, len(train_cams) - 1)]
            lab = cam_k.objects if cam_k.objects is not None else None
            if lab is None:
                from PIL import Image
                import numpy as np
                p = os.path.join(masks_dir, cam_k.image_name + ".png")
                if not os.path.exists(p):
                    continue
                m = np.array(Image.open(p)).astype(np.int64)
                if m.shape != (cam_k.image_height, cam_k.image_width):
                    m = np.array(Image.fromarray(m.astype(np.int32)).resize(
                        (cam_k.image_width, cam_k.image_height), Image.NEAREST))
                lab = torch.from_numpy(m.astype(np.int64)).cuda()
            out_k = render(cam_k, gaussians, pipe, background, use_identity=True)
            id_maps.append(out_k["render_object"])   # (16,H,W)
            lab_maps.append(lab)

        loss_2d = torch.tensor(0.0, device="cuda")
        if len(id_maps) >= 2:
            loss_2d = multi_positive_contrastive_loss(
                id_maps, head, lab_maps,
                temperature=s2["temperature"], pixels_per_view=s2["pixels_per_view"])

        # ---- 3D KL regularization (paper 3.5.4, Eq. 8) ----
        loss_3d = torch.tensor(0.0, device="cuda")
        if iteration % s2["reg3d_interval"] == 0:
            ident = gaussians._identity_dc.squeeze(1)              # (N,16)
            prob = torch.softmax(head(ident), dim=-1)              # (N,256)
            loss_3d = kl_regularization_3d(
                gaussians.get_xyz.detach(), prob,
                k=s2["reg3d_k"], sample_size=s2["reg3d_sample_size"],
                max_points=s2["reg3d_max_points"])

        loss = L_recon + s2["lambda_2d"] * loss_2d + s2["lambda_3d"] * loss_3d
        loss.backward()

        with torch.no_grad():
            ema = 0.4 * loss.item() + 0.6 * ema
            if iteration % 10 == 0:
                progress.set_postfix({"Loss": "%.6f" % ema,
                                      "L2d": "%.4f" % float(loss_2d),
                                      "L3d": "%.4f" % float(loss_3d)})

            if iteration < s2["iterations"]:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

            if iteration in cfg["log"]["test_iterations"] and len(scene.getTestCameras()):
                torch.cuda.empty_cache()
                ps = 0.0
                for tcam in scene.getTestCameras()[:10]:
                    with torch.no_grad():
                        im = render(tcam, gaussians, pipe, background, use_identity=False)["render"]
                        ps += psnr(im, tcam.original_image.cuda()).mean().double()
                print("\n[iter %d] test PSNR %.2f" % (iteration, ps / min(10, len(scene.getTestCameras()))))
                torch.cuda.empty_cache()

            if iteration in cfg["log"]["save_iterations"] or iteration == s2["iterations"]:
                def _save_extra(out):
                    torch.save(head.state_dict(), os.path.join(out, "embed_head.pth"))
                scene.save(iteration, extra=_save_extra)
                print("\n[iter %d] saved" % iteration)

    print("\nStage-2 training complete.")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    set_seed(args.seed)
    training(load_config(args.config))
