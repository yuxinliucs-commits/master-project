import torch
from mcggs.utils.graphics_utils import getWorld2View2, getProjectionMatrix


class Camera(torch.nn.Module):
    def __init__(self, uid, R, T, FoVx, FoVy, image, image_name, gt_mask=None,
                 trans=torch.tensor([0.0, 0.0, 0.0]), scale=1.0, data_device="cuda",
                 depth_image=None):
        super().__init__()
        self.uid = uid
        self.image_name = image_name
        try:
            self.data_device = torch.device(data_device)
        except Exception:
            self.data_device = torch.device("cpu")

        self.R = torch.as_tensor(R, dtype=torch.float32)
        self.T = torch.as_tensor(T, dtype=torch.float32)
        self.FoVx = FoVx
        self.FoVy = FoVy

        self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        if gt_mask is not None:
            self.objects = gt_mask.to(torch.int64).to(self.data_device)
        else:
            self.objects = None

        if depth_image is not None:
            self.depth = depth_image.to(self.data_device)
        else:
            self.depth = None

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = getWorld2View2(R, T, trans, scale).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(
            znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy
        ).transpose(0, 1).cuda()
        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))
        ).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

    def to(self, device):
        self.original_image = self.original_image.to(device)
        if self.objects is not None:
            self.objects = self.objects.to(device)
        if self.depth is not None:
            self.depth = self.depth.to(device)
        return self
