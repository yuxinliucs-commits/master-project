import os
import torch


def recompose_objects(gaussians, group_ids, group_a, group_b, out_path=None):
    """Scene recomposition: exchange the 3D positions of two Gaussian groups (paper Sec. 3.6b).
    Only positions change; shape/appearance untouched, no parameter tuning."""
    mask_a = gaussians.group_mask(group_ids, group_a)
    mask_b = gaussians.group_mask(group_ids, group_b)
    gaussians.swap_group_positions(mask_a, mask_b)
    if out_path:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        gaussians.save_ply(out_path)
    return int(mask_a.sum()), int(mask_b.sum())
