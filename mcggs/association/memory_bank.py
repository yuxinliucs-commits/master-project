import torch


class MemoryBank:
    """3D perceptual memory bank (paper Sec. 3.5.1).

    Stores one group of Gaussian indices per scene instance. A mask from a new view is
    matched by Gaussian overlap (IOCUR form), which is independent of bank size and thus
    avoids frequent threshold retuning (overlap threshold fixed at 0.1 in the paper)."""

    def __init__(self, device="cuda"):
        self.device = device
        self.groups = []            # list of 1-D LongTensors of Gaussian indices
        self.assigned = torch.zeros(0, dtype=torch.long, device=device)  # every Gaussian joins at most one group

    @property
    def num_groups(self):
        return len(self.groups)

    def initialize(self, front_gaussians):
        """First image: every mask forms a distinct group; group ID = mask label (paper 3.5.1)."""
        self.groups = []
        self.assigned = torch.zeros(0, dtype=torch.long, device=self.device)
        for g in front_gaussians:
            self._append_group(g)
        return torch.arange(self.num_groups, dtype=torch.long, device=self.device)

    def _append_group(self, gauss_idx):
        gauss_idx = torch.unique(gauss_idx)
        self.groups.append(gauss_idx)
        self.assigned = torch.unique(torch.cat([self.assigned, gauss_idx]))

    def _extend_group(self, group_idx, gauss_idx):
        gauss_idx = torch.unique(gauss_idx)
        new_members = gauss_idx[~torch.isin(gauss_idx, self.assigned)]
        if new_members.numel() == 0:
            return
        self.groups[group_idx] = torch.unique(torch.cat([self.groups[group_idx], new_members]))
        self.assigned = torch.unique(torch.cat([self.assigned, new_members]))

    @staticmethod
    def overlap(front_gaussians, group):
        """IOCUR-style overlap (paper Eq. 5): |shared| / (|front| + |shared|)."""
        shared = torch.isin(front_gaussians, group).sum().item()
        denom = front_gaussians.numel() + shared
        return shared / max(denom, 1)

    def match(self, front_gaussians, overlap_threshold=0.1, topk=3):
        """Find best group by overlap. Returns (group_idx, overlap, topk candidates list)."""
        if self.num_groups == 0:
            return -1, 0.0, []
        overlaps = [self.overlap(front_gaussians, g) for g in self.groups]
        order = torch.argsort(torch.tensor(overlaps, device=self.device), descending=True)[:topk]
        candidates = [(int(i), overlaps[int(i)]) for i in order]
        best = candidates[0]
        return best[0], best[1], candidates

    def update(self, group_idx, front_gaussians):
        if group_idx < 0 or group_idx >= self.num_groups:
            self._append_group(front_gaussians)
            return self.num_groups - 1
        self._extend_group(group_idx, front_gaussians)
        return group_idx
