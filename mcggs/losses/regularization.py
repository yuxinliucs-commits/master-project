import torch


def kl_regularization_3d(xyz, prob_embed, k=5, sample_size=1000, max_points=300000):
    """3D regularization loss (paper Sec. 3.5.4, Eq. 8, adopted from Gaussian Grouping).

    xyz:        (N,3) Gaussian centers.
    prob_embed: (N,C) softmax probabilities of the embedding head applied to identity encodings.
    k:          number of nearest neighbors (paper: 5).
    sample_size:n sampled Gaussians per step (paper: 1000).

    For each sampled Gaussian i, its encoding distribution is pulled toward those of its
    k nearest neighbors via KL(p_i || p_j)."""
    with torch.no_grad():
        N = xyz.shape[0]
        if N > max_points:
            stride = max(1, N // max_points)
            xyz = xyz[::stride]
            prob_embed = prob_embed[::stride]
            N = xyz.shape[0]
        n = min(sample_size, N)
        sample_idx = torch.randint(0, N, (n,), device=xyz.device)

        sampled = xyz[sample_idx]
        d2 = torch.cdist(sampled, xyz) ** 2                       # (n,N)
        d2[torch.arange(n, device=xyz.device), sample_idx] = float("inf")
        _, nn_idx = d2.topk(k, largest=False)                     # (n,k)

    p_i = prob_embed[sample_idx]                                  # (n,C)
    p_j = prob_embed[nn_idx]                                      # (n,k,C)
    kl = (p_i.unsqueeze(1) *
          (torch.log(p_i.clamp(min=1e-8)).unsqueeze(1) -
           torch.log(p_j.clamp(min=1e-8)))).sum(dim=-1)           # (n,k)
    return kl.mean()
