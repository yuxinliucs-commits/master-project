import os
import numpy as np
import torch
from plyfile import PlyData, PlyElement

from mcggs.utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from mcggs.utils.sh_utils import RGB2SH
from mcggs.utils.graphics_utils import BasicPointCloud
from mcggs.utils.loss_utils import image_gradient, laplacian
from mcggs.scene.cameras import Camera


def _distCUDA2(points):
    """Mean squared distance to 3 nearest neighbors via simple-knn; torch fallback."""
    try:
        from simple_knn._C import distCUDA2
        return distCUDA2(points)
    except Exception:
        with torch.no_grad():
            d2 = torch.cdist(points, points) ** 2
            d2.fill_diagonal_(float("inf"))
            return d2.topk(3, largest=False).values.mean(dim=1)


class GroupedGaussianModel(torch.nn.Module):
    """3DGS with a per-Gaussian identity encoding (paper Sec. 4.1: 16-dim) and
    budget-controlled, score-guided densification (paper Sec. 3.7)."""

    def setup_functions(self):
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.covariance_activation = self.build_covariance_from_scaling_rotation
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize

    @staticmethod
    def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
        L = build_rotation(rotation) @ torch.diag_embed(scaling)
        actual_covariance = L @ L.transpose(1, 2)
        symm = actual_covariance
        return symm

    def __init__(self, sh_degree: int, identity_dim: int = 16):
        super().__init__()
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree
        self.identity_dim = identity_dim

        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self._identity_dc = torch.empty(0)   # (N,1,identity_dim), paper Sec. 4.1

        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.xyz_gradient_accum_abs = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.geometry_frozen = False
        self.setup_functions()

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        return torch.cat((self._features_dc, self._features_rest), dim=1)

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    @property
    def get_identity(self):
        """Per-Gaussian identity encoding passed to the rasterizer as sh_objs (N,1,C)."""
        return self._identity_dc

    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    @torch.no_grad()
    def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float):
        self.spatial_lr_scale = spatial_lr_scale
        if not isinstance(pcd.points, torch.Tensor):
            points = torch.tensor(np.asarray(pcd.points), dtype=torch.float32)
            colors = torch.tensor(np.asarray(pcd.colors), dtype=torch.float32)
        else:
            points, colors = pcd.points.float(), pcd.colors.float()
        fused_point_cloud = points.cuda()
        fused_color = RGB2SH(colors).cuda()
        N = fused_point_cloud.shape[0]
        features = torch.zeros((N, 3, (self.max_sh_degree + 1) ** 2), device="cuda")
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0

        dist2 = _distCUDA2(fused_point_cloud).clamp(min=1e-7)
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
        rots = torch.zeros((N, 4), device="cuda")
        rots[:, 0] = 1
        opacities = inverse_sigmoid(0.1 * torch.ones((N, 1), device="cuda"))
        identity = torch.randn((N, 1, self.identity_dim), device="cuda") * 0.01

        self._xyz = torch.nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = torch.nn.Parameter(features[:, :, :1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = torch.nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = torch.nn.Parameter(scales.requires_grad_(True))
        self._rotation = torch.nn.Parameter(rots.requires_grad_(True))
        self._opacity = torch.nn.Parameter(opacities.requires_grad_(True))
        self._identity_dc = torch.nn.Parameter(identity.requires_grad_(True))
        self.max_radii2D = torch.zeros((N,), device="cuda")
        self.xyz_gradient_accum = torch.zeros((N, 1), device="cuda")
        self.xyz_gradient_accum_abs = torch.zeros((N, 1), device="cuda")
        self.denom = torch.zeros((N, 1), device="cuda")

    def freeze_geometry(self):
        """Stage-2 mode: only identity encoding and the embedding head stay trainable (paper 4.1)."""
        self.geometry_frozen = True
        for p in [self._xyz, self._features_dc, self._features_rest,
                  self._scaling, self._rotation, self._opacity]:
            p.requires_grad_(False)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def training_setup(self, cfg, stage: str, embed_head=None):
        self.percent_dense = cfg["stage1"]["percent_dense"] if stage == "stage1" else 0.01
        s1 = cfg["stage1"]
        s2 = cfg["stage2"]
        self.xyz_schedule = get_expon_lr_func(
            lr_init=s1["position_lr_init"] * self.spatial_lr_scale,
            lr_final=s1["position_lr_final"] * self.spatial_lr_scale,
            lr_delay_steps=48,
            lr_delay_mult=0.01,
            max_steps=s1["position_lr_max_steps"])

        if stage == "stage1":
            l = [
                {"params": [self._xyz], "lr": s1["position_lr_init"] * self.spatial_lr_scale, "name": "xyz"},
                {"params": [self._features_dc], "lr": s1["feature_lr"], "name": "f_dc"},
                {"params": [self._features_rest], "lr": s1["feature_lr"] / 20.0, "name": "f_rest"},
                {"params": [self._opacity], "lr": s1["opacity_lr"], "name": "opacity"},
                {"params": [self._scaling], "lr": s1["scaling_lr"], "name": "scaling"},
                {"params": [self._rotation], "lr": s1["rotation_lr"], "name": "rotation"},
            ]
        else:
            l = [
                {"params": [self._identity_dc], "lr": s2["identity_lr"], "name": "identity"},
            ]
            if embed_head is not None:
                l.append({"params": embed_head.parameters(), "lr": s2["embed_lr"], "name": "embed"})
        self.optimizer = torch.optim.Adam(l, lr=0.0)
        self.optimizer_groups = l
        self.embed_head_ref = embed_head

    def update_learning_rate(self, iteration):
        for pg in self.optimizer.param_groups:
            if pg["name"] == "xyz" and not self.geometry_frozen:
                lr = self.xyz_schedule(iteration)
                pg["lr"] = lr
                return lr
        return 0.0

    # ------------------------------------------------------------------
    # Densification (paper Sec. 3.7: budget schedule + score-based sampling)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def budget_schedule(self, start_count, budget, densify_from_iter, densify_until_iter, interval):
        """Parabolic growth schedule from `start_count` to `budget` (Eq. 2 style, Taming-3DGS)."""
        num_steps = max((densify_until_iter - densify_from_iter) // interval, 1)
        slope_lb = (budget - start_count) / num_steps
        k = 2 * slope_lb
        a = (budget - start_count - k * num_steps) / (num_steps * num_steps)
        values = [int(a * (x ** 2) + k * x + start_count) for x in range(num_steps)]
        return values

    @torch.no_grad()
    def compute_importance_scores(self, cameras, render_fn, bg, weights,
                                  sample_views=20, max_points=300000):
        """Combined score of image saliency (L1+Laplacian) and per-Gaussian attributes (Sec. 3.7)."""
        N = self._xyz.shape[0]
        view_count = torch.zeros(N, device="cuda")
        grad_acc = self.xyz_gradient_accum.abs().squeeze(-1) / self.denom.clamp(min=1)
        opacity = self.get_opacity.squeeze(-1)
        scale_norm = self.get_scaling.norm(dim=-1)
        depth_sum = torch.zeros(N, device="cuda")
        blend_sum = torch.zeros(N, device="cuda")
        saliency_proj = torch.zeros(N, device="cuda")

        views = cameras[:sample_views] if len(cameras) > sample_views else cameras
        for cam in views:
            out = render_fn(cam, self, pipe_stub, bg)
            gt = cam.original_image.cuda()
            loss_map = (out["render"] - gt).abs().mean(dim=0, keepdim=True)
            edge_map = laplacian(gt).abs()
            saliency = loss_map + edge_map
            # accumulate per-Gaussian saliency through blending weights returned by render_fn
            if "blend_weights" in out and out["blend_weights"] is not None:
                idxs, w = out["blend_weights"]          # flattened pixel->gaussian contributions
                per_g = torch.zeros(N, device="cuda")
                sal_flat = saliency.flatten()[idxs[:, 1]]
                per_g.scatter_add_(0, idxs[:, 0], sal_flat * w)
                saliency_proj += per_g
                depth_sum.scatter_add_(0, idxs[:, 0],
                                       cam.depth.flatten()[idxs[:, 1]] * w if cam.depth is not None else w)
                blend_sum.scatter_add_(0, idxs[:, 0], w)
            view_count[out["visibility_filter"]] += 1

        if torch.any(blend_sum > 0):
            depth_score = torch.where(blend_sum > 0, depth_sum / blend_sum.clamp(min=1e-6),
                                      torch.zeros_like(depth_sum))
            sal_score = saliency_proj
        else:
            # projection-only fallback if the rasterizer does not expose blending weights
            sal_score = view_count * opacity
            depth_score = view_count

        score = (weights.get("view_importance", 50) * torch.log1p(view_count) +
                 weights.get("grad_importance", 25) * torch.log1p(grad_acc * 1e4) +
                 weights.get("opac_importance", 100) * opacity +
                 weights.get("scale_importance", 25) * torch.log1p(scale_norm) +
                 weights.get("dept_importance", 5) * torch.log1p(depth_score) +
                 weights.get("blend_importance", 50) * torch.log1p(blend_sum) +
                 weights.get("mse_importance", 50) * torch.log1p(sal_score))
        return score

    @torch.no_grad()
    def densify_by_score(self, target_count, scores):
        """Add Gaussians by score-weighted sampling (clone or split) until target is met."""
        current = self._xyz.shape[0]
        need = target_count - current
        if need <= 0:
            return 0
        probs = (scores - scores.min() + 1e-8)
        probs = probs / probs.sum()
        n_split = int(min(need * 0.5, (self._xyz.shape[0] * 0.1) + 1))
        n_clone = need - n_split
        if n_clone > 0:
            idx_clone = torch.multinomial(probs, min(n_clone, probs.numel()), replacement=True)
            self._clone_tensor(idx_clone)
        if n_split > 0:
            idx_split = torch.multinomial(probs, min(n_split, probs.numel()), replacement=True)
            self._split_tensor(idx_split)
        return need

    @torch.no_grad()
    def _clone_tensor(self, idx):
        new_identity = self._identity_dc[idx]
        self._xyz = torch.nn.Parameter(torch.cat([self._xyz.data, self._xyz.data[idx]], dim=0))
        self._features_dc = torch.nn.Parameter(torch.cat([self._features_dc.data, self._features_dc.data[idx]], dim=0))
        self._features_rest = torch.nn.Parameter(torch.cat([self._features_rest.data, self._features_rest.data[idx]], dim=0))
        self._scaling = torch.nn.Parameter(torch.cat([self._scaling.data, self._scaling.data[idx]], dim=0))
        self._rotation = torch.nn.Parameter(torch.cat([self._rotation.data, self._rotation.data[idx]], dim=0))
        self._opacity = torch.nn.Parameter(torch.cat([self._opacity.data, self._opacity.data[idx]], dim=0))
        # identity encodings are inherited by densified Gaussians (paper Sec. 5.1)
        self._identity_dc = torch.nn.Parameter(torch.cat([self._identity_dc.data, new_identity], dim=0))
        self._rebuild_state()

    @torch.no_grad()
    def _split_tensor(self, idx):
        stds = self.get_scaling[idx].repeat(2, 1)
        stds = torch.log(stds / 1.6)
        rots = build_rotation(self._rotation[idx]).repeat(2, 1, 1)
        samples = torch.einsum("nij,nj->ni", rots, torch.randn_like(self._xyz[idx].repeat(2, 1)) * stds.exp())
        new_xyz = self._xyz.data[idx].repeat(2, 1) + samples
        for name in ["_xyz", "_features_dc", "_features_rest", "_scaling", "_rotation",
                     "_opacity", "_identity_dc"]:
            src = getattr(self, name).data
            dup = src[idx].repeat(2, *[1] * (src.dim() - 1)) if name != "_xyz" else new_xyz
            if name == "_scaling":
                dup = torch.log(stds) if False else src[idx].repeat(2, 1)  # keep log-space; refined by opt
            setattr(self, name, torch.nn.Parameter(torch.cat([src, dup], dim=0)))
        self._rebuild_state()

    @torch.no_grad()
    def _prune(self, prune_mask):
        valid = ~prune_mask
        for name in ["_xyz", "_features_dc", "_features_rest", "_scaling", "_rotation",
                     "_opacity", "_identity_dc", "max_radii2D", "xyz_gradient_accum",
                     "xyz_gradient_accum_abs", "denom"]:
            t = getattr(self, name)
            if isinstance(t, torch.nn.Parameter):
                setattr(self, name, torch.nn.Parameter(t.data[valid]))
            else:
                setattr(self, name, t[valid])

    @torch.no_grad()
    def _rebuild_state(self):
        N = self._xyz.shape[0]
        self.max_radii2D = torch.zeros(N, device="cuda")
        self.xyz_gradient_accum = torch.zeros(N, 1, device="cuda")
        self.xyz_gradient_accum_abs = torch.zeros(N, 1, device="cuda")
        self.denom = torch.zeros(N, 1, device="cuda")

    @torch.no_grad()
    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(
            viewspace_point_tensor.grad[update_filter, :2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1

    @torch.no_grad()
    def standard_prune(self, max_radii2D_threshold=100, opacity_threshold=0.005,
                       size_threshold=None, cameras_extent=None):
        prune_mask = (self.max_radii2D > max_radii2D_threshold)
        prune_mask = prune_mask | (self.get_opacity < opacity_threshold).squeeze(-1)
        if size_threshold is not None:
            big = torch.exp(self._scaling).max(dim=1).values > size_threshold
            prune_mask = prune_mask | big
        self._prune(prune_mask)

    # ------------------------------------------------------------------
    # Group operations (paper Sec. 3.6)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def group_mask(self, group_ids, target):
        """Boolean mask of Gaussians whose label is in `target` (id or list)."""
        labels = group_ids  # (N,) int64
        if not isinstance(target, (list, tuple, torch.Tensor)):
            target = [target]
        t = torch.as_tensor(target, device=labels.device)
        return torch.isin(labels, t)

    @torch.no_grad()
    def remove_gaussians(self, keep_mask):
        self._prune(~keep_mask)

    @torch.no_grad()
    def swap_group_positions(self, mask_a, mask_b):
        """Scene recomposition: exchange 3D positions of two groups (paper Sec. 3.6b)."""
        ca, cb = self._xyz.data[mask_a].mean(0), self._xyz.data[mask_b].mean(0)
        self._xyz.data[mask_a] += (cb - ca)
        self._xyz.data[mask_b] += (ca - cb)

    @torch.no_grad()
    def colorize_group(self, mask, rgb01):
        """Only SH DC coefficients change (paper Sec. 3.6d). rgb01: (3,) in [0,1]."""
        from mcggs.utils.sh_utils import RGB2SH
        sh = RGB2SH(torch.as_tensor(rgb01, dtype=torch.float32, device=self._xyz.device))
        self._features_dc.data[mask, 0, :] = sh

    # ------------------------------------------------------------------
    # IO
    # ------------------------------------------------------------------
    def capture(self):
        return (self.active_sh_degree, self._xyz, self._features_dc, self._features_rest,
                self._scaling, self._rotation, self._opacity, self._identity_dc,
                self.max_radii2D, self.xyz_gradient_accum, self.denom, self.optimizer.state_dict(),
                self.spatial_lr_scale)

    def restore(self, model_args):
        (self.active_sh_degree, self._xyz, self._features_dc, self._features_rest,
         self._scaling, self._rotation, self._opacity, self._identity_dc,
         self.max_radii2D, self.xyz_gradient_accum, self.denom, opt_state,
         self.spatial_lr_scale) = model_args
        if self.optimizer is not None:
            self.optimizer.load_state_dict(opt_state)

    def save_ply(self, path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        xyz = self._xyz.detach().cpu().numpy()
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).cpu().numpy()
        normals = np.zeros_like(xyz)
        shs = np.concatenate([f_dc, f_rest], axis=1)
        identity = self._identity_dc.detach().reshape(-1, self.identity_dim).cpu().numpy()
        opacities = inverse_sigmoid(self.get_opacity.detach()).cpu().numpy()
        scale0 = self.get_scaling.detach().cpu().numpy()
        rot = self.get_rotation.detach().cpu().numpy()

        dtype = [("x", "f4"), ("y", "f4"), ("z", "f4"), ("nx", "f4"), ("ny", "f4"), ("nz", "f4")]
        for i in range(shs.shape[1]):
            dtype.append(("f_rest_%d" % i, "f4"))
        for i in range(self.identity_dim):
            dtype.append(("identity_%d" % i, "f4"))
        dtype += [("opacity", "f4")]
        for i in range(3):
            dtype.append(("scale_%d" % i, "f4"))
        for i in range(4):
            dtype.append(("rot_%d" % i, "f4"))
        elements = np.empty(xyz.shape[0], dtype=dtype)
        elements["x"], elements["y"], elements["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        elements["nx"], elements["ny"], elements["nz"] = normals[:, 0], normals[:, 1], normals[:, 2]
        for i in range(shs.shape[1]):
            elements["f_rest_%d" % i] = shs[:, i]
        for i in range(self.identity_dim):
            elements["identity_%d" % i] = identity[:, i]
        elements["opacity"] = opacities[:, 0]
        for i in range(3):
            elements["scale_%d" % i] = scale0[:, i]
        for i in range(4):
            elements["rot_%d" % i] = rot[:, i]
        PlyData([PlyElement.describe(elements, "vertex")], text=False).write(path)

    def load_ply(self, path):
        ply = PlyData.read(path)
        v = ply["vertex"]
        self.active_sh_degree = self.max_sh_degree
        self._xyz = torch.nn.Parameter(torch.tensor(
            np.stack([v["x"], v["y"], v["z"]], axis=1), dtype=torch.float32).cuda().requires_grad_(True))
        n_sh = 3 * ((self.max_sh_degree + 1) ** 2)
        sh = torch.tensor(np.stack([v["f_rest_%d" % i] for i in range(n_sh)], axis=1),
                          dtype=torch.float32).cuda()
        self._features_dc = torch.nn.Parameter(sh[:, :3].reshape(-1, 1, 3).requires_grad_(True))
        self._features_rest = torch.nn.Parameter(sh[:, 3:].reshape(-1, (self.max_sh_degree + 1) ** 2 - 1, 3)
                                                 .requires_grad_(True))
        self._scaling = torch.nn.Parameter(torch.tensor(
            np.stack([v["scale_%d" % i] for i in range(3)], axis=1), dtype=torch.float32).cuda()
            .requires_grad_(True))
        self._rotation = torch.nn.Parameter(torch.tensor(
            np.stack([v["rot_%d" % i] for i in range(4)], axis=1), dtype=torch.float32).cuda()
            .requires_grad_(True))
        self._opacity = torch.nn.Parameter(torch.tensor(
            v["opacity"], dtype=torch.float32).reshape(-1, 1).cuda().requires_grad_(True))
        ident = torch.tensor(np.stack([v["identity_%d" % i] for i in range(self.identity_dim)], axis=1),
                             dtype=torch.float32).cuda()
        self._identity_dc = torch.nn.Parameter(ident.reshape(-1, 1, self.identity_dim).requires_grad_(True))
        self._rebuild_state()
