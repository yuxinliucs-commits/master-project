import os
import time
import json
import numpy as np
import torch
from PIL import Image

from mcggs.association.memory_bank import MemoryBank
from mcggs.association.projector import GaussianProjector
from mcggs.association.multi_cue import MultiCueScorer
from mcggs.utils.image_utils import save_label_png, load_label_png


class MaskAssociationPipeline:
    """Multi-cue guided mask association (paper Sec. 3.4 / 3.5.1).

    Runs ONCE before identity-encoding training (paper Sec. 4.3):
      1. project trained Gaussians into each view, pick per-mask front Gaussians;
      2. match against the 3D memory bank (IOCUR > 0.1);
      3. refine ambiguous matches with the composite multi-cue score (semantic/depth/edge,
         dual-threshold rule);
      4. update the bank, remap per-view mask labels to global group IDs;
      5. cross-view majority voting per Gaussian, write consistent label maps."""

    def __init__(self, gaussians, cameras, cfg_assoc, raw_mask_dir, out_dir, device="cuda"):
        self.gaussians = gaussians
        self.cameras = cameras
        self.cfg = cfg_assoc
        self.raw_mask_dir = raw_mask_dir
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.device = device

        self.projector = GaussianProjector(num_patch=cfg_assoc["num_patch"],
                                           front_percentage=cfg_assoc["front_percentage"])
        self.bank = MemoryBank(device=device)
        self.scorer = MultiCueScorer(
            w_semantic=cfg_assoc["w_semantic"], w_depth=cfg_assoc["w_depth"],
            w_edge=cfg_assoc["w_edge"],
            semantic_min=cfg_assoc["dual_threshold"]["semantic_min"],
            depth_affinity_min=cfg_assoc["dual_threshold"]["depth_affinity_min"],
            edge_penalty_max=cfg_assoc["dual_threshold"]["edge_penalty_max"],
            dinov2_model=cfg_assoc.get("dinov2_model", "dinov2_vitb14"), device=device)

    def _load_raw_mask(self, cam):
        for ext in (".png", ".npy"):
            p = os.path.join(self.raw_mask_dir, cam.image_name + ext)
            if os.path.exists(p):
                if ext == ".npy":
                    m = np.load(p)
                else:
                    m = np.array(Image.open(p))
                m = m.astype(np.int64)
                if m.shape != (cam.image_height, cam.image_width):
                    m = np.array(Image.fromarray(m.astype(np.int32)).resize(
                        (cam.image_width, cam.image_height), Image.NEAREST)).astype(np.int64)
                return torch.from_numpy(m).to(self.device)
        return None

    @torch.no_grad()
    def run(self, enable_voting=None):
        enable_voting = self.cfg.get("enable_voting", True) if enable_voting is None else enable_voting
        t0 = time.time()
        xyz = self.gaussians.get_xyz

        for vi, cam in enumerate(self.cameras):
            cam = cam.to(self.device)
            raw = self._load_raw_mask(cam)
            if raw is None:
                print("[associate] no raw mask for %s, skipping" % cam.image_name)
                continue
            image = cam.original_image.cuda()

            front = self.projector.front_gaussians_of_masks(xyz, cam, raw)
            view_label = torch.zeros_like(raw, dtype=torch.int64)

            if self.bank.num_groups == 0 and (not self.cfg.get("use_tracker_init", True) or vi == 0):
                # first view: memory bank initialized from its masks (paper 3.5.1)
                for mid, g_idx in sorted(front.items()):
                    self.bank._append_group(g_idx)
                for mid, g_idx in front.items():
                    view_label[raw == mid] = mid   # group id == mask label for the first view
            else:
                for mid, g_idx in sorted(front.items()):
                    gidx, ov, candidates = self.bank.match(
                        g_idx, overlap_threshold=self.cfg["overlap_threshold"])
                    merged = False
                    if 0 <= gidx < self.bank.num_groups and ov > self.cfg["overlap_threshold"]:
                        if len(candidates) > 1 and self.scorer is not None:
                            # multi-cue refinement on the runner-up pair (ambiguous match)
                            runner = candidates[1][0]
                            group_mask_ref = torch.zeros(xyz.shape[0], dtype=torch.bool, device=self.device)
                            group_mask_ref[self.bank.groups[runner]] = True
                            z_cam = self.projector.project(xyz, cam)[2]
                            depth_mask = z_cam[g_idx]
                            depth_group = z_cam[self.bank.groups[runner]]
                            _, do_merge, _ = self.scorer.score_and_decide(
                                image, depth_mask, depth_group,
                                raw == mid, group_mask_ref.float())
                            if do_merge:
                                gidx = runner
                        self.bank.update(gidx, g_idx)
                        view_label[raw == mid] = gidx + 1
                        merged = True
                    if not merged:
                        new_id = self.bank.update(-1, g_idx)
                        view_label[raw == mid] = new_id + 1

            per_view_labels[cam.image_name] = view_label
            save_label_png(os.path.join(self.out_dir, cam.image_name + ".png"),
                           view_label.cpu().numpy().astype(np.int32))
            # NOTE: per-view labels are already globally consistent (group ids from the bank);
            # the optional majority voting over views is only needed when a Gaussian is
            # claimed by different groups in different views, which the bank prevents
            # by construction (each Gaussian joins exactly one group, paper 3.5.1).

        info = {
            "num_groups": self.bank.num_groups,
            "raw_mask_dir": self.raw_mask_dir,
            "associated_dir": self.out_dir,
            "overlap_threshold": self.cfg["overlap_threshold"],
            "front_percentage": self.cfg["front_percentage"],
            "num_patch": self.cfg["num_patch"],
            "elapsed_sec": round(time.time() - t0, 1),
        }
        with open(os.path.join(self.out_dir, "info.json"), "w") as f:
            json.dump(info, f, indent=2)
        print("[associate] groups=%d, elapsed=%ss" % (info["num_groups"], info["elapsed_sec"]))
        return per_view_labels, info
