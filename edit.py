"""Scene editing entry (paper Sec. 3.6): remove / recompose / inpaint / colorize.
All operations act on group ids of a trained grouped Gaussian field, without retraining."""
import os
from argparse import ArgumentParser

import numpy as np
import torch
from PIL import Image

from mcggs.config import load_config
from mcggs.scene import Scene
from mcggs.scene.gaussian_model import GroupedGaussianModel
from mcggs.editing.removal import remove_object
from mcggs.editing.recompose import recompose_objects
from mcggs.editing.colorize import colorize_object
from mcggs.editing.inpaint import inpaint_object


def load_group_ids(gaussians, cfg, masks_dir, cameras):
    """Derive per-Gaussian group ids by projecting into training views and majority voting
    over the associated (consistent) label maps (paper Sec. 3.5.1 voting)."""
    from mcggs.association.projector import GaussianProjector
    proj = GaussianProjector(num_patch=1, front_percentage=1.0)
    N = gaussians.get_xyz.shape[0]
    votes = []
    counts = []
    with torch.no_grad():
        for cam in cameras:
            p = os.path.join(masks_dir, cam.image_name + ".png")
            if not os.path.exists(p):
                continue
            lab = np.array(Image.open(p)).astype(np.int64)
            if lab.shape != (cam.image_height, cam.image_width):
                lab = np.array(Image.fromarray(lab.astype(np.int32)).resize(
                    (cam.image_width, cam.image_height), Image.NEAREST))
            lab_t = torch.from_numpy(lab).cuda()
            u, v, z, valid = proj.project(gaussians.get_xyz, cam)
            H, W = cam.image_height, cam.image_width
            inside = valid & (u >= 0) & (u < W) & (v >= 0) & (v < H)
            ui, vi = u[inside].long().clamp(0, W - 1), v[inside].long().clamp(0, H - 1)
            labels = lab_t[vi, ui]
            gi = torch.nonzero(inside).squeeze(-1)
            v_t = torch.zeros(N, dtype=torch.long, device="cuda")
            c_t = torch.zeros(N, dtype=torch.long, device="cuda")
            v_t[gi] = labels
            c_t[gi] = 1
            votes.append(v_t)
            counts.append(c_t)
    assert votes, "no associated masks found in %s" % masks_dir
    votes = torch.stack(votes, dim=0).float()
    counts = torch.stack(counts, dim=0).float().sum(0)
    votes = votes.sum(0)
    group_ids = torch.where(counts > 0, (votes / counts.clamp(min=1)).round().long(),
                            torch.zeros_like(votes, dtype=torch.long))
    return group_ids


def main():
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--operation", type=str, required=True,
                        choices=["remove", "recompose", "inpaint", "colorize"])
    parser.add_argument("--target", type=int, default=None, help="target group id")
    parser.add_argument("--target_b", type=int, default=None, help="second group id (recompose)")
    parser.add_argument("--color", type=float, nargs=3, default=None, help="rgb in [0,1] (colorize)")
    parser.add_argument("--out_name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    torch.manual_seed(args.seed)

    cfg = load_config(args.config)
    data = cfg["data"]
    gaussians = GroupedGaussianModel(data["sh_degree"], identity_dim=cfg["stage2"]["identity_dim"])
    scene = Scene(cfg, gaussians, load_iteration=-1, shuffle=False)
    group_ids = load_group_ids(gaussians, cfg,
                               os.path.join(data["source_path"], data["masks_dir"]),
                               scene.getTrainCameras())
    out_name = args.out_name or ("%s_%s" % (args.operation, args.target))
    out_path = os.path.join(data["model_path"], "edited", out_name, "point_cloud.ply")

    if args.operation == "remove":
        n = remove_object(gaussians, group_ids, args.target, out_path)
        print("[remove] deleted %d gaussians of group %d" % (n, args.target))
    elif args.operation == "recompose":
        assert args.target_b is not None, "--target_b required for recompose"
        na, nb = recompose_objects(gaussians, group_ids, args.target, args.target_b, out_path)
        print("[recompose] swapped positions of %d and %d gaussians" % (na, nb))
    elif args.operation == "colorize":
        assert args.color is not None, "--color r g b required for colorize"
        n = colorize_object(gaussians, group_ids, args.target, tuple(args.color), out_path)
        print("[colorize] recolored %d gaussians of group %d" % (n, args.target))
    elif args.operation == "inpaint":
        # first delete the target, then repair the exposed region (paper 4.5.2)
        remove_object(gaussians, group_ids, args.target, None)
        bg = torch.tensor([1, 1, 1] if data["white_background"] else [0, 0, 0],
                          dtype=torch.float32, device="cuda")
        n_new = inpaint_object(gaussians, scene.getTrainCameras(), cfg["pipeline"], bg,
                               cfg["editing"], out_path=out_path)
        print("[inpaint] removed group %d and added %d new gaussians" % (args.target, n_new))

    print("[edit] saved to", out_path)
    print("[edit] render with: python render.py --splits both "
          "(after pointing data.model_path at the edited folder if desired)")


if __name__ == "__main__":
    main()
