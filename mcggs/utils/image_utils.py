import torch
import os
from PIL import Image
import numpy as np


def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


def imread_uint8(path):
    return np.array(Image.open(path).convert("RGB"))


def imwrite(path, arr):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    Image.fromarray(arr).save(path)


def load_label_png(path):
    """Load a label mask (uint16 capable) as a numpy int32 array."""
    arr = np.array(Image.open(path))
    return arr.astype(np.int32)


def save_label_png(path, labels):
    """Save label ids; use uint16 when the id space exceeds 255."""
    max_id = int(labels.max()) if labels.size else 0
    if max_id > 255:
        Image.fromarray(labels.astype(np.uint16)).save(path)
    else:
        Image.fromarray(labels.astype(np.uint8)).save(path)


def colorize_labels(labels, num_colors=4096, seed=0):
    """Map label ids to a random-color visualization (H,W,3 uint8)."""
    rng = np.random.RandomState(seed)
    palette = rng.randint(0, 255, size=(num_colors, 3), dtype=np.uint8)
    palette[0] = 0
    out = np.zeros((*labels.shape, 3), dtype=np.uint8)
    ids = labels.astype(np.int64)
    np.clip(ids, 0, num_colors - 1, out=ids)
    out = palette[ids]
    out[labels == 0] = 0
    return out
