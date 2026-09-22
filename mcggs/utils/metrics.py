import torch
import numpy as np

try:
    import lpips as _lpips
    _LPIPS = _lpips.LPIPS(net="vgg")
    for p in _LPIPS.parameters():
        p.requires_grad = False

    def lpips_distance(img1, img2):
        """imgs: (3,H,W) in [0,1]."""
        with torch.no_grad():
            a = (img1.clamp(0, 1) * 2 - 1).unsqueeze(0).cuda()
            b = (img2.clamp(0, 1) * 2 - 1).unsqueeze(0).cuda()
            return _LPIPS(a, b).item()
except Exception:
    def lpips_distance(img1, img2):
        raise ImportError("pip install lpips is required for LPIPS evaluation.")


def best_assignment_iou(pred_labels, gt_labels, iou_threshold=None):
    """IoU between predicted instance labels and GT panoptic labels, ignoring class info,
    aligned by best linear assignment (paper Sec. 4.1/4.4). Returns (mean_iou_percent, detail)."""
    from scipy.optimize import linear_sum_assignment

    pred = pred_labels.astype(np.int64).ravel()
    gt = gt_labels.astype(np.int64).ravel()
    pred_ids = np.unique(pred)
    pred_ids = pred_ids[pred_ids > 0]
    gt_ids = np.unique(gt)
    gt_ids = gt_ids[gt_ids > 0]
    if len(pred_ids) == 0 or len(gt_ids) == 0:
        return 0.0, {}

    pred_index = {v: i for i, v in enumerate(pred_ids)}
    gt_index = {v: i for i, v in enumerate(gt_ids)}
    inter = np.zeros((len(pred_ids), len(gt_ids)), dtype=np.int64)
    for p_id, g_id in zip(pred, gt):
        if p_id > 0 and g_id > 0:
            inter[pred_index[p_id], gt_index[g_id]] += 1
    pred_area = inter.sum(axis=1, keepdims=True)
    gt_area = inter.sum(axis=0, keepdims=True)
    union = pred_area + gt_area - inter
    iou = np.where(union > 0, inter / np.maximum(union, 1), 0.0)

    rows, cols = linear_sum_assignment(-iou)
    matched = {int(pred_ids[r]): (int(gt_ids[c]), float(iou[r, c])) for r, c in zip(rows, cols)}
    per_pred_iou = [matched[int(p)][1] for p in pred_ids]
    mean_iou = float(np.mean(per_pred_iou)) * 100.0
    if iou_threshold is not None:
        prec = float(np.mean([iou >= iou_threshold for iou in per_pred_iou])) * 100.0
        matched_gt = set(g for g, _ in matched.values())
        rec = float(np.mean([any(m[0] == g and m[1] >= iou_threshold for m in matched.values())
                             for g in gt_ids])) * 100.0
        return mean_iou, {"precision@%.2f" % iou_threshold: prec,
                          "recall@%.2f" % iou_threshold: rec,
                          "per_pred": matched}
    return mean_iou, {"per_pred": matched}


def iou_drop(iou_full, iou_sparse):
    """IoU drop rate (paper Sec. 4.4), in percent."""
    if iou_full <= 0:
        return 0.0
    return (iou_full - iou_sparse) / iou_full * 100.0


def _load_clip():
    try:
        import clip
    except Exception:
        raise ImportError("pip install git+https://github.com/openai/CLIP.git for CLIP metrics.")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    return model, preprocess, device, clip


def clip_direction_similarity(orig_img, edited_img, orig_text, edited_text):
    """CLIP Direction Similarity (paper Sec. 4.5.4, Instruct-NeRF2NeRF protocol).
    imgs: PIL.Image or (H,W,3) uint8 arrays."""
    from PIL import Image as PILImage

    model, preprocess, device, clip = _load_clip()

    def to_pil(x):
        if isinstance(x, PILImage.Image):
            return x
        return PILImage.fromarray(np.asarray(x))

    with torch.no_grad():
        img_feats = []
        for x in (to_pil(orig_img), to_pil(edited_img)):
            t = preprocess(x).unsqueeze(0).to(device)
            img_feats.append(model.encode_image(t))
        img_delta = img_feats[1] - img_feats[0]
        img_delta = img_delta / img_delta.norm(dim=-1, keepdim=True)

        txt = clip.tokenize([orig_text, edited_text]).to(device)
        txt_feats = model.encode_text(txt)
        txt_delta = txt_feats[1] - txt_feats[0]
        txt_delta = txt_delta / txt_delta.norm(dim=-1, keepdim=True)

    return float((img_delta @ txt_delta.T).item())


def load_dinov2(model_name="dinov2_vitb14", device="cuda"):
    """DINOv2 feature extractor (paper Sec. 3.5.1 semantic cue). Returns (model, transform_size)."""
    try:
        model = torch.hub.load("facebookresearch/dinov2", model_name)
    except Exception as e:
        raise ImportError("Could not load DINOv2 via torch.hub: %s" % e)
    model.eval().to(device)
    return model
