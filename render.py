"""Render RGB and/or identity label maps from a trained grouped Gaussian field."""
import os
from argparse import ArgumentParser

import numpy as np
import torch
from PIL import Image

from mcggs.config import load_config
from mcggs.scene import Scene
from mcggs.scene.gaussian_model import GroupedGaussianModel
from mcggs.gaussian_renderer import render
from mcggs.utils.image_utils import colorize_labels, imwrite


def main():
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--mode", type=str, default="both", choices=["rgb", "identity", "both"])
    parser.add_argument("--splits", type=str, default="test", choices=["test", "train", "both"])
    args = parser.parse_args()
    cfg = load_config(args.config)
    data = cfg["data"]

    out_dir = args.out_dir or os.path.join(data["model_path"], "render")
    gaussians = GroupedGaussianModel(data["sh_degree"], identity_dim=cfg["stage2"]["identity_dim"])
    scene = Scene(cfg, gaussians, load_iteration=args.iteration, shuffle=False)
    bg = torch.tensor([1, 1, 1] if data["white_background"] else [0, 0, 0],
                      dtype=torch.float32, device="cuda")

    cam_sets = []
    if args.splits in ("test", "both"):
        cam_sets += [("test", c) for c in scene.getTestCameras()]
    if args.splits in ("train", "both"):
        cam_sets += [("train", c) for c in scene.getTrainCameras()]

    with torch.no_grad():
        for split, cam in cam_sets:
            out = render(cam, gaussians, cfg["pipeline"], bg,
                         use_identity=args.mode in ("identity", "both"))
            rgb = (out["render"].clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            imwrite(os.path.join(out_dir, split, "rgb", cam.image_name + ".png"), rgb)
            if out["render_object"] is not None:
                ident = out["render_object"]                     # (16,H,W)
                labels = ident.argmax(dim=0).cpu().numpy().astype(np.int32)
                imwrite(os.path.join(out_dir, split, "identity_label", cam.image_name + ".png"),
                        labels)
                imwrite(os.path.join(out_dir, split, "identity_vis", cam.image_name + ".png"),
                        colorize_labels(labels))
    print("[render] wrote results to", out_dir)


if __name__ == "__main__":
    main()
