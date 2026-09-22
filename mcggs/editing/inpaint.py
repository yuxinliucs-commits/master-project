import os
import numpy as np
import torch
from tqdm import tqdm

from mcggs.gaussian_renderer import render, render_visibility_counts
from mcggs.utils.general_utils import inverse_sigmoid
from mcggs.utils.sh_utils import RGB2SH


def detect_holes(viewpoint_cameras, gaussians):
    """After deletion, project the remaining scene onto every training view; pixels with no
    valid Gaussian contribution are marked as holes (paper Sec. 4.5.2: only the regions
    invisible across the views are inpainted, never the whole 2D silhouette)."""
    counts = render_visibility_counts(viewpoint_cameras, gaussians)
    return [cnt == 0 for cnt in counts]


def _seed_gaussians_from_holes(cam, hole_mask, n_points, device="cuda"):
    """Back-project a subsample of a view's hole pixels into 3D to seed new Gaussians."""
    H, W = hole_mask.shape
    ys, xs = torch.nonzero(hole_mask, as_tuple=True)
    if ys.numel() == 0:
        return None
    if ys.numel() > n_points:
        sel = torch.randperm(ys.numel(), device=device)[:n_points]
        ys, xs = ys[sel], xs[sel]
    focal_x = cam.image_width / (2 * torch.tan(torch.tensor(cam.FoVx / 2, device=device)))
    focal_y = cam.image_height / (2 * torch.tan(torch.tensor(cam.FoVy / 2, device=device)))
    depth = 0.6 * float(cam.depth.median()) if cam.depth is not None else 2.0
    dir_cam = torch.stack([(xs - W / 2) / focal_x, (ys - H / 2) / focal_y,
                           torch.ones_like(xs, dtype=torch.float32)], dim=1)
    W2C = cam.world_view_transform.t()
    R = W2C[:3, :3]
    c = W2C[3, :3]
    dir_world = dir_cam @ R
    xyz = c[None, :] + depth * dir_world
    rgb = torch.rand(xyz.shape[0], 3, device=device) * 0.6 + 0.2
    return xyz, rgb


class _LamaFallback:
    """Without a LaMa environment, the original view is used as the optimization target.
    Note (paper Sec. 4.5.2): with real LaMa inpainting the restored region keeps high-frequency
    texture; this fallback merely fills the hole with observed background."""

    def __call__(self, image, hole_mask):
        return image


def inpaint_object(gaussians, viewpoint_cameras, pipe, bg, cfg_edit, device="cuda",
                   lama_runner=None, out_path=None):
    """Object inpainting (paper Sec. 3.6c / 4.5.2):
      1. the target Gaussians must already be deleted;
      2. detect per-view pixels with no Gaussian contribution (holes);
      3. obtain 2D inpainting targets (LaMa if available, else fallback);
      4. initialize new Gaussians seeded from the holes and optimize ONLY them with a
         photometric loss; original Gaussians stay frozen, so the shared 3D representation
         keeps the restored region view-consistent."""
    holes = detect_holes(viewpoint_cameras, gaussians)

    # 2D targets per view (LaMa result when provided)
    lama_runner = lama_runner or _LamaFallback()
    targets = []
    for cam, hole in zip(viewpoint_cameras, holes):
        img = cam.original_image.cuda()
        if isinstance(lama_runner, _LamaFallback):
            targets.append(img)
        else:
            arr = (img.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            out = lama_runner(arr, hole.cpu().numpy())
            targets.append(torch.from_numpy(np.asarray(out)).float().permute(2, 0, 1).cuda() / 255.0)

    # seed new Gaussians from every view's holes
    all_xyz, all_rgb = [], []
    n_per_view = max(int(cfg_edit.get("inpaint_new_gaussians", 20000)) // max(len(viewpoint_cameras), 1), 100)
    for cam, hole in zip(viewpoint_cameras, holes):
        seeded = _seed_gaussians_from_holes(cam, hole, n_per_view, device)
        if seeded is not None:
            all_xyz.append(seeded[0])
            all_rgb.append(seeded[1])
    if not all_xyz:
        print("[inpaint] no holes detected; nothing to inpaint")
        return 0
    new_xyz = torch.cat(all_xyz, dim=0)
    new_rgb = torch.cat(all_rgb, dim=0)
    N = new_xyz.shape[0]

    dist2 = torch.cdist(new_xyz, new_xyz).clamp(min=1e-6).topk(4, largest=False).values[:, 1:].mean(1) ** 2
    scale_log = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
    params = {
        "xyz": torch.nn.Parameter(new_xyz.clone()),
        "f_dc": torch.nn.Parameter(RGB2SH(new_rgb).unsqueeze(1).clone()),
        "opacity": torch.nn.Parameter(inverse_sigmoid(0.1 * torch.ones(N, 1, device=device))),
        "scaling": torch.nn.Parameter(scale_log),
        "rotation": torch.nn.Parameter(torch.cat(
            [torch.ones(N, 1, device=device), torch.zeros(N, 3, device=device)], dim=1)),
    }
    opt = torch.optim.Adam([
        {"params": [params["xyz"]], "lr": cfg_edit.get("inpaint_lr_xyz", 0.00016)},
        {"params": [params["f_dc"]], "lr": cfg_edit.get("inpaint_lr_features", 0.0025)},
        {"params": [params["opacity"]], "lr": cfg_edit.get("inpaint_lr_opacity", 0.05)},
        {"params": [params["scaling"]], "lr": cfg_edit.get("inpaint_lr_scaling", 0.005)},
        {"params": [params["rotation"]], "lr": cfg_edit.get("inpaint_lr_rotation", 0.001)},
    ])

    names = ["_xyz", "_features_dc", "_features_rest", "_scaling", "_rotation", "_opacity", "_identity_dc"]
    iterations = int(cfg_edit.get("inpaint_iterations", 3000))
    progress = tqdm(range(iterations), desc="inpaint")
    for it in progress:
        base = gaussians
        zeros_f_rest = torch.zeros(N, base._features_rest.shape[1], 3, device=device)
        zeros_ident = torch.zeros(N, 1, base.identity_dim, device=device)
        cat = {
            "_xyz": torch.cat([base._xyz.data, params["xyz"]]),
            "_features_dc": torch.cat([base._features_dc.data, params["f_dc"]]),
            "_features_rest": torch.cat([base._features_rest.data, zeros_f_rest]),
            "_scaling": torch.cat([base._scaling.data, params["scaling"]]),
            "_rotation": torch.cat([base._rotation.data, params["rotation"]]),
            "_opacity": torch.cat([base._opacity.data, params["opacity"]]),
            "_identity_dc": torch.cat([base._identity_dc.data, zeros_ident]),
        }
        orig = {k: getattr(base, k) for k in names}
        for k in names:
            setattr(base, k, torch.nn.Parameter(cat[k]))

        cam = viewpoint_cameras[it % len(viewpoint_cameras)].to(device)
        out = render(cam, base, pipe, bg, use_identity=False)
        loss = (out["render"] - targets[it % len(targets)]).abs().mean()
        opt.zero_grad()
        loss.backward()
        opt.step()

        for k in names:
            setattr(base, k, orig[k])
        if it % 100 == 0:
            progress.set_postfix({"loss": "%.4f" % loss.item()})

    # merge the optimized new Gaussians into the scene permanently
    def cat_param(name, new_t):
        setattr(gaussians, name, torch.nn.Parameter(
            torch.cat([getattr(gaussians, name).data.detach(), new_t.detach()], dim=0)))
    zeros_f_rest = torch.zeros(N, gaussians._features_rest.shape[1], 3, device=device)
    cat_param("_xyz", params["xyz"])
    cat_param("_features_dc", params["f_dc"])
    cat_param("_features_rest", zeros_f_rest)
    cat_param("_scaling", params["scaling"])
    cat_param("_rotation", params["rotation"])
    cat_param("_opacity", params["opacity"])
    cat_param("_identity_dc", torch.zeros(N, 1, gaussians.identity_dim, device=device))
    gaussians._rebuild_state()

    if out_path:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        gaussians.save_ply(out_path)
    return N
