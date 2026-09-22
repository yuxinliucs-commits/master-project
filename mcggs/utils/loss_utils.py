import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp


def l1_loss(network_output, gt):
    return torch.abs(network_output - gt).mean()


def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def ssim(img1, img2, window_size=11, size_average=True):
    channels = 1
    img1 = torch.unsqueeze(img1, 0) if img1.dim() == 3 else img1
    img2 = torch.unsqueeze(img2, 0) if img2.dim() == 3 else img2
    channels = img1.shape[1]
    window = create_window(window_size, channels)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channels)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channels)

    mu1_sq, mu2_sq, mu1_mu2 = mu1.pow(2), mu2.pow(2), mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channels) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channels) - mu1_mu2

    C1, C2 = 0.01 ** 2, 0.03 ** 2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean() if size_average else ssim_map.mean(1).mean(1).mean(1)


def image_gradient(img):
    """Spatial gradient magnitude of an (3,H,W) or (1,H,W) image tensor."""
    gray = img.mean(dim=0, keepdim=True) if img.dim() == 3 else img
    gx = gray[:, :, 1:] - gray[:, :, :-1]
    gy = gray[:, 1:, :] - gray[:, :-1, :]
    g = torch.zeros_like(gray)
    g[:, :, 1:] += gx.abs()
    g[:, 1:, :] += gy.abs()
    return g


def laplacian(img):
    """Laplacian filter response of an (1,H,W) or (3,H,W) image tensor (high-frequency cue)."""
    gray = img.mean(dim=0, keepdim=True) if img.dim() == 3 else img
    kernel = torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
                          device=gray.device, dtype=gray.dtype).view(1, 1, 3, 3)
    return F.conv2d(gray.unsqueeze(0), kernel, padding=1).squeeze(0)
