import torch
import torch.nn.functional as F


class MultiCueScorer:
    """Composite merge score (paper Sec. 3.5.1, Eq. 6; weights from Table 4.5:
    w = 0.625 / 0.25 / 0.125) and the dual-threshold merge rule.

    S(mask, group) = w_sem * cosine(DINOv2 means) + w_dep * depth_affinity - w_edge * boundary_penalty
    merge  iff  semantic > semantic_min  and  depth_affinity > depth_affinity_min
                and  boundary_penalty < edge_penalty_max."""

    def __init__(self, w_semantic=0.625, w_depth=0.25, w_edge=0.125,
                 semantic_min=0.5, depth_affinity_min=0.5, edge_penalty_max=0.5,
                 dinov2_model="dinov2_vitb14", device="cuda"):
        self.w_semantic = w_semantic
        self.w_depth = w_depth
        self.w_edge = w_edge
        self.semantic_min = semantic_min
        self.depth_affinity_min = depth_affinity_min
        self.edge_penalty_max = edge_penalty_max
        self.device = device
        self._dinov2 = None
        self._dinov2_name = dinov2_model

    # ---------------- semantic cue ----------------
    def _get_dinov2(self):
        if self._dinov2 is None:
            from mcggs.utils.metrics import load_dinov2
            self._dinov2 = load_dinov2(self._dinov2_name, device=self.device)
        return self._dinov2

    @torch.no_grad()
    def dinov2_feature(self, image, mask=None):
        """Average DINOv2 feature of the (optionally masked) image region.
        image: (3,H,W) in [0,1]; mask: (H,W) bool."""
        model = self._get_dinov2()
        x = image.unsqueeze(0)
        if mask is not None:
            m = mask.unsqueeze(0).unsqueeze(0).float()
            x = x * m
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        if mask is not None:
            m = F.interpolate(m, size=(224, 224), mode="nearest")
        feats = model.forward_features(x) if hasattr(model, "forward_features") else model(x)
        if feats.dim() == 4:
            feats = feats[:, 1:]                     # drop cls token for ViT with tokens
        else:
            feats = feats[:, 1:] if feats.shape[1] > 1 else feats
        if mask is not None:
            m_flat = m.flatten(2).squeeze(0).squeeze(0).bool()
            if m_flat.sum() > 0:
                feats = feats[:, m_flat]
        return feats.mean(dim=1).squeeze(0)

    def semantic_similarity(self, image, mask_a, mask_b):
        try:
            fa = self.dinov2_feature(image, mask_a > 0)
            fb = self.dinov2_feature(image, mask_b > 0)
            sim = F.cosine_similarity(fa[None], fb[None]).item()
            return (sim + 1.0) / 2.0                 # map [-1,1] -> [0,1]
        except ImportError:
            return None                              # cue disabled gracefully

    # ---------------- depth cue ----------------
    @staticmethod
    def depth_affinity(depth_a, depth_b):
        """exp(-|median diff| / (0.5*max(median)+eps)); depths are median Gaussian z of each mask."""
        da = torch.median(depth_a).item() if depth_a.numel() else 0.0
        db = torch.median(depth_b).item() if depth_b.numel() else 0.0
        scale = 0.5 * max(abs(da), abs(db)) + 1e-6
        return float(torch.exp(torch.tensor(-abs(da - db) / scale)).item())

    # ---------------- edge cue ----------------
    @staticmethod
    def boundary_penalty(image, mask_a, mask_b, dilation=5):
        """Normalized mean gradient magnitude inside the boundary band between the two masks."""
        from mcggs.utils.loss_utils import image_gradient
        grad = image_gradient(image)                 # (1,H,W)
        ma = (mask_a > 0).float()[None, None]
        mb = (mask_b > 0).float()[None, None]
        k = 2 * dilation + 1
        kernel = torch.ones(1, 1, k, k, device=image.device)
        dil = F.conv2d(mb, kernel, padding=dilation) > 0
        band = (F.conv2d(ma, kernel, padding=dilation) > 0).float() * dil.float() * (1 - ma)
        denom = band.sum()
        if denom.item() < 1:
            return 0.0
        return (grad * band.squeeze(0)).sum().item() / denom.item()

    # ---------------- composite ----------------
    def score_and_decide(self, image, depth_a, depth_b, mask_a, mask_b):
        """Returns (score, merge_bool, parts dict)."""
        sem = self.semantic_similarity(image, mask_a, mask_b)
        sem_available = sem is not None
        sem = sem if sem_available else 0.0
        dep = self.depth_affinity(depth_a, depth_b)
        edge = self.boundary_penalty(image, mask_a, mask_b)
        score = (self.w_semantic * sem + self.w_depth * dep - self.w_edge * edge)
        if sem_available:
            merge = (sem > self.semantic_min and dep > self.depth_affinity_min
                     and edge < self.edge_penalty_max)
        else:
            # without DINOv2 fall back to depth/edge cues only
            merge = (dep > self.depth_affinity_min and edge < self.edge_penalty_max)
        return score, merge, {"semantic": sem, "depth": dep, "edge": edge}
