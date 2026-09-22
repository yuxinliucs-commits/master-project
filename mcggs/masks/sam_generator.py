import os
import numpy as np
import torch
from PIL import Image

from mcggs.utils.image_utils import save_label_png


def generate_sam_masks(source_path, images_dir, out_dir, conf_threshold=0.5,
                       points_per_side=32, pred_iou_thresh=0.86, stability_score_thresh=0.92,
                       sam_checkpoint=None, model_type="vit_h", device="cuda"):
    """Per-view class-agnostic SAM masks (paper Sec. 3.3 / 4.1).

    Masks are sorted by confidence; those scoring below `conf_threshold` (0.5) are discarded
    (paper Sec. 4.1). Output: one uint16/uint8 label PNG per image (0 = background) under
    `out_dir`, plus a metadata JSON. Requires `pip install segment-anything` and a checkpoint."""
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

    os.makedirs(out_dir, exist_ok=True)
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint).to(device)
    generator = SamAutomaticMaskGenerator(
        sam, points_per_side=points_per_side, pred_iou_thresh=pred_iou_thresh,
        stability_score_thresh=stability_score_thresh)

    img_root = os.path.join(source_path, images_dir)
    names = sorted([f for f in os.listdir(img_root)
                    if f.lower().endswith((".png", ".jpg", ".jpeg"))])
    meta = {}
    for fname in names:
        name = os.path.splitext(fname)[0]
        img = np.array(Image.open(os.path.join(img_root, fname)).convert("RGB"))
        masks = generator.generate(img)
        masks = sorted(masks, key=lambda m: m.get("score", 0.0), reverse=True)
        labels = np.zeros(img.shape[:2], dtype=np.int32)
        kept = []
        next_id = 1
        for m in masks:
            score = m.get("score", 1.0)
            if score < conf_threshold:
                continue
            region = m["segmentation"]
            labels[region] = next_id
            kept.append({"id": next_id, "score": float(score),
                         "area": int(m.get("area", region.sum()))})
            next_id += 1
        save_label_png(os.path.join(out_dir, name + ".png"), labels)
        meta[name] = kept
        print("[sam] %s: %d masks kept" % (name, len(kept)))
    with open(os.path.join(out_dir, "raw_mask_meta.json"), "w") as f:
        json.dump(meta, f)
    return out_dir


def validate_raw_mask_dir(raw_mask_dir, image_names=None):
    """Check that a raw mask folder (e.g. produced by SAM or copied from the reference
    repositories' raw_sam_mask) covers the expected views."""
    if not os.path.isdir(raw_mask_dir):
        return False, "folder does not exist: %s" % raw_mask_dir
    files = [f for f in os.listdir(raw_mask_dir) if f.endswith((".png", ".npy"))]
    if not files:
        return False, "no label images found"
    if image_names:
        missing = [n for n in image_names if not any(f.startswith(n) for f in files)]
        if missing:
            return False, "missing masks for %d views, e.g. %s" % (len(missing), missing[:3])
    return True, "%d mask files" % len(files)
