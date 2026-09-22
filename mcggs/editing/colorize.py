import os
import torch


def colorize_object(gaussians, group_ids, target, rgb01, out_path=None):
    """Object colourisation: only the SH DC coefficients of the target group change
    (paper Sec. 3.6d); shape, texture details and shading are preserved."""
    mask = gaussians.group_mask(group_ids, target)
    gaussians.colorize_group(mask, rgb01)
    if out_path:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        gaussians.save_ply(out_path)
    return int(mask.sum())
