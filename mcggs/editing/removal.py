import os
import torch


def remove_object(gaussians, group_ids, target, out_path=None):
    """Object removal: delete every Gaussian of the target group (paper Sec. 3.6a).
    Single deletion pass, no optimization (paper Sec. 4.5.1)."""
    keep = ~gaussians.group_mask(group_ids, target)
    n_removed = int((~keep).sum().item())
    gaussians.remove_gaussians(keep)
    if out_path:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        gaussians.save_ply(out_path)
    return n_removed
