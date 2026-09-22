import torch
import math
from typing import NamedTuple


class BasicPointCloud(NamedTuple):
    points: torch.Tensor
    colors: torch.Tensor
    normals: torch.Tensor


def getWorld2View(R, t):
    Rt = torch.zeros((4, 4), device=R.device if isinstance(R, torch.Tensor) else "cpu")
    Rt[:3, :3] = torch.as_tensor(R, dtype=torch.float32)
    Rt[:3, 3] = torch.as_tensor(t, dtype=torch.float32)
    Rt[3, 3] = 1.0
    return Rt


def getWorld2View2(R, t, trans=torch.tensor([0.0, 0.0, 0.0]), scale=1.0):
    if not isinstance(R, torch.Tensor):
        R = torch.tensor(R, dtype=torch.float32)
    if not isinstance(t, torch.Tensor):
        t = torch.tensor(t, dtype=torch.float32)
    if not isinstance(trans, torch.Tensor):
        trans = torch.tensor(trans, dtype=torch.float32)
    Rt = torch.zeros((4, 4))
    Rt[:3, :3] = R.t()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    C2W = torch.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center - trans) / scale
    C2W[:3, 3] = cam_center
    Rt = torch.linalg.inv(C2W)
    return Rt


def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovX = math.tan((fovX / 2))
    tanHalfFovY = math.tan((fovY / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)
    z_sign = 1.0
    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))


def focal2fov(focal, pixels):
    return 2 * math.atan(pixels / (2 * focal))


def ndc2Pixel(v, S):
    return ((v + 1.0) * S - 1.0) * 0.5
