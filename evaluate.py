"""Evaluation (paper Sec. 4.1-4.5):
  render : PSNR / SSIM / LPIPS on test views (Sec. 4.2.1)
  seg    : instance IoU with best linear assignment vs GT panoptic labels (Sec. 4.1/4.4)
  clip   : CLIP direction similarity for edits (Sec. 4.5.4)
  iou_drop : IoU drop between full and sparse training-view models (Sec. 4.4)"""
import os
from argparse import ArgumentParser

import numpy as np
import torch

from mcggs.config import load_config
from mcggs.scene import Scene
from mcggs.scene.gaussian_model import GroupedGaussianModel
from mcggs.gaussian_renderer import render
from mcggs.utils.image_utils import psnr
from mcggs.utils.loss_utils import ssim
from mcggs.utils.metrics import best_assignment_iou, iou_drop, lpips_distance, clip_direction_similarity


def eval_render(cfg, scene, gaussians, bg):
    pipe = cfg["pipeline"]
    ps, ss, lp = [], [], []
    with torch.no_grad():
        for cam in scene.getTestCameras():
            out = render(cam, gaussians, pipe, bg, use_identity=False)
            im = out["render"].clamp(0, 1)
            gt = cam.original_image.cuda().clamp(0, 1)
            ps.append(psnr(im, gt).mean().item())
            ss.append(ssim(im, gt).item())
            try:
                lp.append(lpips_distance(im, gt))
            except ImportError:
                pass
    res = {"PSNR": float(np.mean(ps)), "SSIM": float(np.mean(ss))}
    if lp:
        res["LPIPS"] = float(np.mean(lp))
    print("[render]", res)
    return res


def eval_segmentation(cfg, scene, gaussians, bg, gt_label_dir):
    from PIL import Image
    pipe = cfg["pipeline"]
    ious = []
    for cam in scene.getTestCameras():
        p = os.path.join(gt_label_dir, cam.image_name + ".png")
        if not os.path.exists(p):
            print("[seg] no GT label for", cam.image_name)
            continue
        with torch.no_grad():
            out = render(cam, gaussians, pipe, bg, use_identity=True)
        labels = out["render_object"].argmax(dim=0).cpu().numpy().astype(np.int32)
        gt = np.array(Image.open(p)).astype(np.int32)
        if gt.shape != labels.shape:
            gt = np.array(Image.fromarray(gt).resize(
                (labels.shape[1], labels.shape[0]), Image.NEAREST))
        iou, _ = best_assignment_iou(labels, gt)
        ious.append(iou)
    mean_iou = float(np.mean(ious)) if ious else 0.0
    print("[seg] IoU (best linear assignment): %.2f%%" % mean_iou)
    return mean_iou


def eval_iou_drop(iou_full, iou_sparse):
    drop = iou_drop(iou_full, iou_sparse)
    print("[seg] IoU drop: %.2f%%" % drop)
    return drop


def eval_clip(cfg, orig_img, edited_img, orig_text, edited_text):
    d = clip_direction_similarity(orig_img, edited_img, orig_text, edited_text)
    print("[clip] direction similarity: %.3f" % d)
    return d


def main():
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--mode", type=str, default="render",
                        choices=["render", "seg", "clip", "iou_drop"])
    parser.add_argument("--gt_label_dir", type=str, default=None,
                        help="GT panoptic label images (seg mode), same names as views")
    parser.add_argument("--iou_full", type=float, default=None, help="for --mode iou_drop")
    parser.add_argument("--iou_sparse", type=float, default=None)
    parser.add_argument("--orig_image", type=str, default=None)
    parser.add_argument("--edited_image", type=str, default=None)
    parser.add_argument("--orig_text", type=str, default=None)
    parser.add_argument("--edited_text", type=str, default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)

    if args.mode == "iou_drop":
        assert args.iou_full is not None and args.iou_sparse is not None
        eval_iou_drop(args.iou_full, args.iou_sparse)
        return
    if args.mode == "clip":
        assert all(x is not None for x in
                   (args.orig_image, args.edited_image, args.orig_text, args.edited_text))
        eval_clip(cfg, args.orig_image, args.edited_image, args.orig_text, args.edited_text)
        return

    data = cfg["data"]
    gaussians = GroupedGaussianModel(data["sh_degree"], identity_dim=cfg["stage2"]["identity_dim"])
    scene = Scene(cfg, gaussians, load_iteration=-1, shuffle=False)
    bg = torch.tensor([1, 1, 1] if data["white_background"] else [0, 0, 0],
                      dtype=torch.float32, device="cuda")
    if args.mode == "render":
        eval_render(cfg, scene, gaussians, bg)
    elif args.mode == "seg":
        assert args.gt_label_dir, "--gt_label_dir required for seg mode"
        eval_segmentation(cfg, scene, gaussians, bg, args.gt_label_dir)


if __name__ == "__main__":
    main()
