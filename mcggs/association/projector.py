import torch


class GaussianProjector:
    """Project Gaussian centers into each view and pick per-mask front Gaussians
    (patch-based, depth-front subset; paper Sec. 3.5.1)."""

    def __init__(self, num_patch=32, front_percentage=0.2):
        self.num_patch = num_patch
        self.front_percentage = front_percentage

    @staticmethod
    def project(xyz, camera):
        """Return (pixel_u (N,), pixel_v (N,), cam_z (N,), valid (N,))."""
        ones = torch.ones(xyz.shape[0], 1, device=xyz.device)
        p_hom = torch.cat([xyz, ones], dim=1) @ camera.full_proj_transform
        w = p_hom[:, 3:].clamp(min=1e-8)
        p_ndc = p_hom[:, :3] / w
        H, W = camera.image_height, camera.image_width
        u = ((p_ndc[:, 0] + 1.0) * W - 1.0) * 0.5
        v = ((p_ndc[:, 1] + 1.0) * H - 1.0) * 0.5
        valid = (p_hom[:, 3] > 0.01) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        return u, v, p_hom[:, 2], valid

    def front_gaussians_of_masks(self, xyz, camera, label_map):
        """For every mask id in `label_map` (H,W int64, 0=background), select the
        depth-front `front_percentage` Gaussians inside each 32x32 patch of the mask.

        Returns dict {mask_id: LongTensor of Gaussian indices}."""
        u, v, z, valid = self.project(xyz, camera)
        H, W = camera.image_height, camera.image_width
        ui = u.long().clamp(0, W - 1)
        vi = v.long().clamp(0, H - 1)
        pid = vi * W + ui

        result = {}
        ids = torch.unique(label_map)
        ids = ids[ids > 0]
        if ids.numel() == 0:
            return result

        mask_flat = label_map.flatten()
        for mid in ids.tolist():
            in_mask_pixel = (mask_flat[pid] == mid)
            hit = valid & in_mask_pixel
            g_idx = torch.nonzero(hit).squeeze(-1)
            if g_idx.numel() == 0:
                continue
            # patch-based front selection: within each patch keep the nearest fraction
            patch_w = max(W // self.num_patch, 1)
            patch_h = max(H // self.num_patch, 1)
            patch_id = (vi[g_idx] // patch_h) * self.num_patch + (ui[g_idx] // patch_w)
            keep = []
            z_sel = z[g_idx]
            for p_id in torch.unique(patch_id):
                members = torch.nonzero(patch_id == p_id).squeeze(-1)
                k = max(int(members.numel() * self.front_percentage), 1)
                nearest = members[torch.argsort(z_sel[members])[:k]]
                keep.append(nearest)
            keep = torch.cat(keep)
            result[mid] = g_idx[keep]
        return result
