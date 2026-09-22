"""Scene readers for COLMAP captures and Blender-style transforms.json."""
import os
import numpy as np
import torch
from PIL import Image

from mcggs.scene.cameras import Camera
from mcggs.utils.graphics_utils import fov2focal, focal2fov
from mcggs.scene import colmap_loader
from mcggs.utils.general_utils import set_seed

CAMERA_EXT = ".png"


def _load_image(path, width, height, resolution):
    img = Image.open(path)
    orig_w, orig_h = img.size
    if resolution in (1, 2, 4, 8):
        scale = resolution
    elif resolution == -1:
        if orig_w > 1600:
            scale = 1600 / orig_w
        else:
            scale = 1.0
    else:
        scale = 1.0
    if scale != 1.0:
        resized_w, resized_h = round(orig_w * scale), round(orig_h * scale)
    else:
        resized_w, resized_h = orig_w, orig_h
    img = img.resize((resized_w, resized_h))
    return torch.from_numpy(np.array(img)).float().permute(2, 0, 1) / 255.0


def _maybe_load_mask(mask_dir, image_name, width, height):
    if not mask_dir:
        return None
    for ext in (".png", ".npy"):
        p = os.path.join(mask_dir, image_name + ext)
        if os.path.exists(p):
            if ext == ".npy":
                m = np.load(p).astype(np.int64)
            else:
                m = np.array(Image.open(p)).astype(np.int64)
            if m.shape != (height, width):
                mi = Image.fromarray(m.astype(np.int32))
                m = np.array(mi.resize((width, height), Image.NEAREST)).astype(np.int64)
            return torch.from_numpy(m.astype(np.int64))
    return None


def _camera_from_colmap(uid, intr, extr, image_path, image_name, resolution, mask_dir, data_device):
    R = np.transpose(extr.R)
    T = np.array(extr.T)
    img = _load_image(image_path, None, None, resolution)
    gt_mask = _maybe_load_mask(mask_dir, image_name, img.shape[2], img.shape[1])
    return Camera(uid=uid, R=R, T=T, FoVx=intr[0], FoVy=intr[1], image=img,
                  image_name=image_name, gt_mask=gt_mask, data_device=data_device)


def readColmapSceneInfo(path, images, resolution, mask_dir, data_device, eval_split, llffhold=8):
    """All cameras are returned; Scene splits train/test with the llffhold rule."""
    sparse_dir = os.path.join(path, "sparse/0")
    cam_extr = colmap_loader.read_images_binary(os.path.join(sparse_dir, "images.bin")) \
        if os.path.exists(os.path.join(sparse_dir, "images.bin")) \
        else colmap_loader.read_images_text(os.path.join(sparse_dir, "images.txt"))
    cam_intr = colmap_loader.read_cameras_binary(os.path.join(sparse_dir, "cameras.bin")) \
        if os.path.exists(os.path.join(sparse_dir, "cameras.bin")) \
        else colmap_loader.read_cameras_text(os.path.join(sparse_dir, "cameras.txt"))
    pts = colmap_loader.read_points3D_binary(os.path.join(sparse_dir, "points3D.bin")) \
        if os.path.exists(os.path.join(sparse_dir, "points3D.bin")) \
        else colmap_loader.read_points3D_text(os.path.join(sparse_dir, "points3D.txt"))
    xyz = np.array([p.xyz for p in pts.values()], dtype=np.float32) if pts else np.zeros((0, 3), np.float32)
    rgb = np.array([p.rgb for p in pts.values()], dtype=np.float32) / 255.0 if pts \
        else np.random.rand(0, 3).astype(np.float32)

    cameras = []
    reading_order = sorted(cam_extr.values(), key=lambda x: x.name)
    for idx, extr in enumerate(reading_order):
        intr = cam_intr[extr.camera_id]
        fovx, fovy = colmap_loader.intrinsics_to_fov(intr)
        image_path = os.path.join(path, images, extr.name)
        name = os.path.splitext(os.path.basename(extr.name))[0]
        cameras.append(_camera_from_colmap(idx, (fovx, fovy), extr, image_path, name,
                                           resolution, mask_dir, data_device))
    return cameras, xyz, rgb


def readNerfSyntheticInfo(path, resolution, mask_dir, data_device, eval_split, white_bkgd=False):
    import json
    with open(os.path.join(path, "transforms_train.json")) as f:
        meta_train = json.load(f)
    meta_test_path = os.path.join(path, "transforms_test.json")
    meta_test = json.load(open(meta_test_path)) if os.path.exists(meta_test_path) else None

    fovx = focal2fov(fov2focal(0.5 * np.pi / 2.0, 800), 800)  # placeholder; recompute per frame below
    cameras = []
    xyz = rgb = None

    def load_set(meta, split):
        cams = []
        for idx, frame in enumerate(meta["frames"]):
            cam_name = os.path.join(path, frame["file_path"] + ".png")
            fovx = frame.get("camera_angle_x")
            if fovx is None:
                continue
            fovy = 2 * np.arctan(np.tan(fovx / 2))
            m = np.array(frame["transform_matrix"])
            m[:3, 1:3] *= -1  # Blender -> COLMAP convention
            m = m[np.array([1, 0, 2, 3]), :]  # swap x/y axes like the standard loader
            R = np.transpose(m[:3, :3])
            T = m[:3, 3]
            img = _load_image(cam_name, None, None, resolution)
            if white_bkgd:
                img[img.abs() < 1e-6] = 1.0
            name = os.path.splitext(os.path.basename(frame["file_path"]))[0]
            gt_mask = _maybe_load_mask(mask_dir, name, img.shape[2], img.shape[1])
            cams.append(Camera(uid=idx, R=R, T=T, FoVx=float(fovx), FoVy=float(fovy),
                               image=img, image_name=name, gt_mask=gt_mask, data_device=data_device))
        return cams

    train_cams = load_set(meta_train, "train")
    test_cams = load_set(meta_test, "test") if meta_test else []
    if not test_cams and eval_split:
        test_cams = [train_cams[i] for i in range(0, len(train_cams), 8)]
        train_cams = [c for i, c in enumerate(train_cams) if i % 8 != 0]
    return train_cams + test_cams, xyz, rgb


def sceneLoadPathCallbacks(source_path, resolution, data_device, eval_split):
    if os.path.exists(os.path.join(source_path, "sparse")):
        return "Colmap", lambda images, mask_dir: readColmapSceneInfo(
            source_path, images, resolution, mask_dir, data_device, eval_split)
    if os.path.exists(os.path.join(source_path, "transforms_train.json")):
        return "Blender", lambda images, mask_dir: readNerfSyntheticInfo(
            source_path, resolution, mask_dir, data_device, eval_split)
    raise RuntimeError("Could not recognize scene type at %s" % source_path)
