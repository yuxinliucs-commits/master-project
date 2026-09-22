import torch


def _sample_labeled_pixels(fmap, lmap, pixels_per_view):
    """fmap: (16,H,W) identity features; lmap: (H,W) int64 labels.
    Returns (emb (M,256) with grad, labels (M,)) after random subsampling."""
    flat16 = fmap.flatten(1).t()                       # (HW,16)
    flat_lab = lmap.flatten().long()                   # (HW,)
    valid = torch.nonzero(flat_lab >= 0).squeeze(-1)
    if valid.numel() == 0:
        return None, None
    if valid.numel() > pixels_per_view:
        sel = valid[torch.randperm(valid.numel(), device=valid.device)[:pixels_per_view]]
    else:
        sel = valid
    emb = torch.nn.functional.normalize(flat16[sel], dim=-1)  # grad flows to embed_head & identity
    return emb, flat_lab[sel]


def _group_centers(emb, labels):
    """Mean embedding per label. Returns (centers (G,256), group_ids (G,))."""
    group_ids = torch.unique(labels)
    centers = torch.stack([emb[labels == g].mean(dim=0) for g in group_ids], dim=0)
    return centers, group_ids


def multi_positive_contrastive_loss(identity_maps, embed_head, label_maps,
                                    temperature=0.1, pixels_per_view=2048,
                                    query_view=0):
    """Multi-positive contrastive loss for identity encodings (paper Sec. 3.5.3).

    identity_maps: list of (16,H,W) rendered identity features, one per sampled view.
    label_maps:    list of (H,W) int64 associated mask labels (label 0 / -1 treated as unlabeled).
    embed_head:    nn.Linear(16, 256) mapping identity encodings to the embedding space.

    For a query pixel of label g in view `query_view`:
      positives P = per-view mean embeddings of label g in the OTHER views (multi-positive);
      negatives D = mean embeddings of every other label across all views.
      L = -(1/|P|) sum_{z+ in P} log( exp(sim(z_q,z+)/tau) / sum_{z' in P+D} exp(sim(z_q,z')/tau) )
    """
    device = identity_maps[0].device
    per_view = []
    for fmap, lmap in zip(identity_maps, label_maps):
        emb, lab = _sample_labeled_pixels(fmap, lmap, pixels_per_view)
        if emb is None:
            continue
        per_view.append((emb, lab))
    if len(per_view) < 2 or query_view >= len(per_view):
        return torch.tensor(0.0, device=device, requires_grad=True)

    q_view_idx = query_view % len(per_view)
    q_emb_all, q_lab_all = per_view[q_view_idx]

    # positive centers per label from the other views
    pos_centers = {}
    other_centers = []
    for i, (emb, lab) in enumerate(per_view):
        if i == q_view_idx:
            continue
        c, g = _group_centers(emb, lab)
        other_centers.append((c, g))
        for gi, ci in zip(g.tolist(), c.unbind(0)):
            pos_centers.setdefault(gi, []).append(ci)

    # negative pool: centers of all other labels (detached for stability)
    neg_pool = torch.cat([c for c, _ in other_centers], dim=0).detach()

    uniq = torch.unique(q_lab_all)
    losses = []
    for g in uniq.tolist():
        if g not in pos_centers:
            continue                                    # no cross-view positive for this label
        sel = (q_lab_all == g)
        zq = q_emb_all[sel].mean(dim=0)                 # (256,) with grad
        P = torch.stack(pos_centers[g], dim=0)          # (P,256)
        # drop negatives that coincide with any positive (same group)
        if neg_pool.numel() > 0:
            sim_to_pos = (neg_pool @ P.t().detach()).max(dim=1).values
            keep = sim_to_pos < 0.999
            D = neg_pool[keep] if keep.any() else neg_pool
        cand = torch.cat([P, D], dim=0) if D is not None and D.numel() > 0 else P
        logits = (cand @ zq) / temperature              # (P+D,) ; positives first
        loss_g = -torch.log_softmax(logits, dim=0)[: P.shape[0]].mean()
        losses.append(loss_g)

    if len(losses) == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)
    return torch.stack(losses).mean()
