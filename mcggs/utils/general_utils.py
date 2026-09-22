import torch


def inverse_sigmoid(x):
    return torch.log(x / (1 - x))


def sigmoid(x):
    return torch.sigmoid(x)


def build_rotation(r):
    norm = torch.sqrt(r[:, 0] * r[:, 0] + r[:, 1] * r[:, 1] + r[:, 2] * r[:, 2] + r[:, 3] * r[:, 3])
    q = r / norm[:, None]
    R = torch.zeros((q.size(0), 3, 3), device=r.device)
    r0, r1, r2, r3 = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R[:, 0, 0] = 1 - 2 * (r2 * r2 + r3 * r3)
    R[:, 0, 1] = 2 * (r1 * r2 - r0 * r3)
    R[:, 0, 2] = 2 * (r1 * r3 + r0 * r2)
    R[:, 1, 0] = 2 * (r1 * r2 + r0 * r3)
    R[:, 1, 1] = 1 - 2 * (r1 * r1 + r3 * r3)
    R[:, 1, 2] = 2 * (r2 * r3 - r0 * r1)
    R[:, 2, 0] = 2 * (r1 * r3 - r0 * r2)
    R[:, 2, 1] = 2 * (r2 * r3 + r0 * r1)
    R[:, 2, 2] = 1 - 2 * (r1 * r1 + r2 * r2)
    return R


def build_scaling_rotation(s, r):
    L = torch.zeros((s.size(0), 3, 3), device=s.device)
    R = build_rotation(r)
    L[:, 0, 0] = s[:, 0]
    L[:, 1, 1] = s[:, 1]
    L[:, 2, 2] = s[:, 2]
    L = R @ L
    return L


def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device=L.device)
    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty


def strip_symmetric(sym):
    return strip_lowerdiag(sym)


def get_expon_lr_func(lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000):
    if lr_delay_steps > 0:
        def warmup_lr_func(step):
            if step < lr_delay_steps:
                return lr_delay_mult + (1 - lr_delay_mult) * step / lr_delay_steps
            return 1.0
    else:
        def warmup_lr_func(step):
            return 1.0

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            return 0.0
        if lr_delay_steps > 0:
            delay_rate = warmup_lr_func(step)
        else:
            delay_rate = 1.0
        if step < lr_delay_steps:
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * step / lr_delay_steps
        else:
            delay_rate = 1.0
        saved_step = step
        step = step - lr_delay_steps
        frac = min(step / max_steps, 1.0) if max_steps > 0 else 1.0
        exp_lrp = lr_init * (lr_final / lr_init) ** frac if lr_init > 0 and lr_final > 0 else lr_final
        return delay_rate * exp_lrp

    return helper


def set_seed(seed):
    import numpy as np
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
