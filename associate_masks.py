"""Multi-cue guided mask association (paper Sec. 3.4 / 3.5.1).
Runs ONCE on a trained stage-1 Gaussian field, before identity-encoding training."""
import os
from argparse import ArgumentParser

import torch

from mcggs.config import load_config
from mcggs.scene import Scene
from mcggs.scene.gaussian_model import GroupedGaussianModel
from mcggs.association.pipeline import MaskAssociationPipeline
from mcggs.utils.general_utils import set_seed


def main():
    parser = ArgumentParser(description="Multi-cue guided mask association (paper 3.5.1)")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--iteration", type=int, default=-1,
                        help="load the latest stage-1 point cloud by default")
    parser.add_argument("--raw_mask_dir", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--no_dinov2", action="store_true",
                        help="disable the semantic cue (fall back to depth/edge only)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    set_seed(args.seed)

    cfg = load_config(args.config)
    data = cfg["data"]
    assoc = dict(cfg["association"])

    raw_mask_dir = args.raw_mask_dir or os.path.join(data["source_path"], "raw_%s_mask" % cfg["masks"]["seg_model"])
    if not os.path.isdir(raw_mask_dir):
        hint = os.path.join(data["model_path"], "raw_mask_dir.txt")
        if os.path.exists(hint):
            raw_mask_dir = open(hint).read().strip()
    assert os.path.isdir(raw_mask_dir), "raw mask folder not found: %s" % raw_mask_dir

    out_dir = args.out_dir or os.path.join(data["source_path"], data["masks_dir"])

    gaussians = GroupedGaussianModel(data["sh_degree"], identity_dim=cfg["stage2"]["identity_dim"])
    scene = Scene(cfg, gaussians, load_iteration=args.iteration, shuffle=False)
    print("[associate] gaussians=%d, train views=%d"
          % (gaussians.get_xyz.shape[0], len(scene.getTrainCameras())))

    if args.no_dinov2:
        assoc["w_semantic"] = 0.0

    pipeline = MaskAssociationPipeline(gaussians, scene.getTrainCameras(), assoc,
                                       raw_mask_dir, out_dir)
    with torch.no_grad():
        pipeline.run()
    print("[associate] associated masks written to", out_dir)


if __name__ == "__main__":
    main()
