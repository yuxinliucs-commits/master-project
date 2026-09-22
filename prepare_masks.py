"""SAM per-view mask generation (paper Sec. 3.3 / 4.1)."""
import os
from argparse import ArgumentParser

from mcggs.config import load_config
from mcggs.masks.sam_generator import generate_sam_masks, validate_raw_mask_dir


def main():
    parser = ArgumentParser(description="Prepare per-view SAM masks (paper 3.3/4.1)")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--sam_checkpoint", type=str, default=None,
                        help="SAM checkpoint path; if omitted, an existing raw mask folder is used")
    parser.add_argument("--model_type", type=str, default="vit_h")
    parser.add_argument("--raw_mask_dir", type=str, default=None,
                        help="use an existing raw mask folder instead of running SAM")
    args = parser.parse_args()
    cfg = load_config(args.config)
    data = cfg["data"]
    masks_cfg = cfg["masks"]

    raw_dir = args.raw_mask_dir or os.path.join(data["source_path"], "raw_%s_mask" % masks_cfg["seg_model"])

    if args.sam_checkpoint:
        generate_sam_masks(
            data["source_path"], data["images"], raw_dir,
            conf_threshold=masks_cfg["confidence_threshold"],
            points_per_side=masks_cfg["points_per_side"],
            pred_iou_thresh=masks_cfg["pred_iou_thresh"],
            stability_score_thresh=masks_cfg["stability_score_thresh"],
            sam_checkpoint=args.sam_checkpoint, model_type=args.model_type)
    else:
        ok, msg = validate_raw_mask_dir(raw_dir)
        if not ok:
            raise SystemExit(
                "Raw mask folder unusable (%s). Provide --sam_checkpoint to run SAM, or\n"
                "point --raw_mask_dir at a folder of per-view label images "
                "(e.g. raw_sam_mask from the reference repositories)." % msg)
        print("[prepare_masks] using existing raw masks: %s (%s)" % (raw_dir, msg))

    # remember the resolved raw mask dir for associate_masks.py
    os.makedirs(data["model_path"], exist_ok=True)
    with open(os.path.join(data["model_path"], "raw_mask_dir.txt"), "w") as f:
        f.write(os.path.abspath(raw_dir))
    print("[prepare_masks] done ->", raw_dir)


if __name__ == "__main__":
    main()
