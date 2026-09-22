import torch


def parabolic_budget_schedule(start_count, budget, densify_from_iter, densify_until_iter,
                              densification_interval):
    """Stable-growth model (paper Sec. 3.7): Gaussian count follows a parabola that starts at
    the initial SfM point count and ends at the user-defined budget.

    Returns a list of target counts, one per densification step."""
    num_steps = max((densify_until_iter - densify_from_iter) // densification_interval, 1)
    slope_lower_bound = (budget - start_count) / num_steps
    k = 2 * slope_lower_bound
    a = (budget - start_count - k * num_steps) / (num_steps * num_steps)
    return [int(a * (x ** 2) + k * x + start_count) for x in range(num_steps)]


def resolve_budget(start_count, budget_cfg):
    """budget_cfg: {'mode': 'multiplier', 'value': 5.0} or {'mode': 'final_count', 'value': 200000}."""
    mode = budget_cfg.get("mode", "multiplier")
    if mode == "multiplier":
        return int(start_count * float(budget_cfg["value"]))
    return int(budget_cfg["value"])


def view_saliency_map(rendered, gt):
    """Per-view saliency combining L1 loss with a Laplacian (high-frequency) filter (Sec. 3.7).
    rendered/gt: (3,H,W). Returns (1,H,W)."""
    from mcggs.utils.loss_utils import laplacian
    l1 = (rendered - gt).abs().mean(dim=0, keepdim=True)
    edge = laplacian(gt).abs()
    if edge.dim() == 2:
        edge = edge.unsqueeze(0)
    return l1 + edge
