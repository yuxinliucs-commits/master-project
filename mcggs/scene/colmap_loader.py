"""Minimal COLMAP binary/text model reader (camera, images, points3D)."""
import os
import struct
import collections
import numpy as np

CameraModel = collections.namedtuple("CameraModel", ["model_id", "model_name", "num_params"])
Camera = collections.namedtuple("Camera", ["id", "model", "width", "height", "params"])
BaseImage = collections.namedtuple(
    "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])
Point3D = collections.namedtuple("Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"])

CAMERA_MODELS = {
    CameraModel(0, "SIMPLE_PINHOLE", 3),
    CameraModel(1, "PINHOLE", 4),
    CameraModel(2, "SIMPLE_RADIAL", 4),
    CameraModel(3, "RADIAL", 5),
    CameraModel(4, "OPENCV", 8),
    CameraModel(5, "OPENCV_FISHEYE", 8),
    CameraModel(6, "FULL_OPENCV", 12),
    CameraModel(7, "FOV", 5),
    CameraModel(8, "SIMPLE_RADIAL_FISHEYE", 4),
    CameraModel(9, "RADIAL_FISHEYE", 5),
    CameraModel(10, "THIN_PRISM_FISHEYE", 12),
}
CAMERA_MODEL_IDS = {m.model_id: m for m in CAMERA_MODELS}


def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2, 2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3], 1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2], 2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2]])


def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)


def read_cameras_binary(path):
    cameras = {}
    with open(path, "rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_cameras):
            props = read_next_bytes(fid, 24, "iiQQ")
            camera_id, model_id, width, height = props[0], props[1], props[2], props[3]
            n_params = CAMERA_MODEL_IDS[model_id].num_params
            params = read_next_bytes(fid, 8 * n_params, "d" * n_params)
            cameras[camera_id] = Camera(id=camera_id, model=CAMERA_MODEL_IDS[model_id].model_name,
                                        width=width, height=height, params=np.array(params))
    return cameras


def read_images_binary(path):
    images = {}
    with open(path, "rb") as fid:
        num_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_images):
            props = read_next_bytes(fid, 64, "idddddddi")
            image_id = props[0]
            qvec = np.array(props[1:5])
            tvec = np.array(props[5:8])
            camera_id = props[8]
            name = ""
            cur = fid.read(1)
            while cur != b"\x00":
                name += cur.decode("utf-8")
                cur = fid.read(1)
            n_points = read_next_bytes(fid, 8, "Q")[0]
            xys = np.concatenate([np.array(read_next_bytes(fid, 8, "dd")) for _ in range(n_points)]) \
                if n_points > 0 else np.zeros((0, 2))
            pids = np.array([read_next_bytes(fid, 4, "i")[0] for _ in range(n_points)]) \
                if n_points > 0 else np.zeros(0, dtype=int)
            images[image_id] = BaseImage(id=image_id, qvec=qvec, tvec=tvec, camera_id=camera_id,
                                         name=name, xys=xys, point3D_ids=pids)
    return images


def read_points3D_binary(path):
    points = {}
    with open(path, "rb") as fid:
        num_points = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_points):
            props = read_next_bytes(fid, 43, "QdddBBBd")
            pid = props[0]
            xyz = np.array(props[1:4])
            rgb = np.array(props[4:7])
            error = props[7]
            track_len = read_next_bytes(fid, 8, "Q")[0]
            track = np.array(read_next_bytes(fid, 12 * track_len, "ii" * track_len)) \
                if track_len > 0 else np.zeros(0)
            points[pid] = Point3D(id=pid, xyz=xyz, rgb=rgb, error=error,
                                  image_ids=track[0::2].astype(int), point2D_idxs=track[1::2].astype(int))
    return points


def read_cameras_text(path):
    cameras = {}
    with open(path, "r") as fid:
        for line in fid:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            el = line.split()
            cameras[int(el[0])] = Camera(id=int(el[0]), model=el[1], width=int(el[2]), height=int(el[3]),
                                         params=np.array(el[4:], dtype=float))
    return cameras


def read_images_text(path):
    images = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line or line.startswith("#"):
                if not line:
                    break
                continue
            el = line.split()
            image_id, qvec, tvec, camera_id, name = (int(el[0]), np.array(el[1:5], float),
                                                     np.array(el[5:8], float), int(el[8]), el[9])
            points_line = fid.readline().split()
            n = int(points_line[0])
            xys = np.array(points_line[1:1 + 2 * n], float).reshape(n, 2) if n else np.zeros((0, 2))
            pids = np.array(points_line[1 + 2 * n:1 + 3 * n], int) if n else np.zeros(0, int)
            images[image_id] = BaseImage(id=image_id, qvec=qvec, tvec=tvec, camera_id=camera_id,
                                         name=name, xys=xys, point3D_ids=pids)
    return images


def read_points3D_text(path):
    points = {}
    with open(path, "r") as fid:
        for line in fid:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            el = line.split()
            pid = int(el[0])
            points[pid] = Point3D(id=pid, xyz=np.array(el[1:4], float), rgb=np.array(el[4:7], int),
                                  error=float(el[7]), image_ids=np.zeros(0), point2D_idxs=np.zeros(0))
    return points


def read_model(path, ext=None):
    if ext is None:
        ext = ".bin" if os.path.exists(os.path.join(path, "cameras.bin")) else ".txt"
    readers = {
        ".bin": (read_cameras_binary, read_images_binary, read_points3D_binary),
        ".txt": (read_cameras_text, read_images_text, read_points3D_text),
    }
    cams_f, imgs_f, pts_f = readers[ext]
    return (cams_f(os.path.join(path, "cameras" + ext)),
            imgs_f(os.path.join(path, "images" + ext)),
            pts_f(os.path.join(path, "points3D" + ext)))


def intrinsics_to_fov(camera):
    """Return (fovX, fovY) for supported camera models."""
    p = camera.params
    if camera.model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL", "SIMPLE_RADIAL_FISHEYE", "RADIAL_FISHEYE"):
        focal = p[0]
    elif camera.model in ("PINHOLE", "OPENCV", "OPENCV_FISHEYE", "FULL_OPENCV", "THIN_PRISM_FISHEYE", "FOV"):
        focal = p[0]
    else:
        raise ValueError("Unsupported camera model: %s" % camera.model)
    from mcggs.utils.graphics_utils import focal2fov
    return focal2fov(focal, camera.width), focal2fov(focal, camera.height)
