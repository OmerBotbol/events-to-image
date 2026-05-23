"""
DRUNet (UNetRes) — Plug-and-Play denoiser for event-based image reconstruction.

Architecture: Zhang et al., "Plug-and-Play Image Restoration with Deep Denoiser Prior",
IEEE TPAMI 2022.  Same architecture used in the TU Berlin event reconstruction repo.

Weights file: models/drunet_gray.pth
Download: https://github.com/cszn/KAIR/releases/download/v1.0/drunet_gray.pth
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── primitive blocks ──────────────────────────────────────────────────────────

def conv(in_channels, out_channels, kernel_size=3, stride=1, padding=1,
         bias=True, mode="CBR", negative_slope=0.2):
    L = []
    for t in mode:
        if t == "C":
            L.append(nn.Conv2d(in_channels, out_channels, kernel_size,
                               stride=stride, padding=padding, bias=bias))
        elif t == "T":
            L.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size,
                                        stride=stride, padding=padding, bias=bias))
        elif t == "B":
            L.append(nn.BatchNorm2d(out_channels, momentum=0.9, eps=1e-04, affine=True))
        elif t == "I":
            L.append(nn.InstanceNorm2d(out_channels, affine=True))
        elif t == "R":
            L.append(nn.ReLU(inplace=True))
        elif t == "r":
            L.append(nn.ReLU(inplace=False))
        elif t == "L":
            L.append(nn.LeakyReLU(negative_slope=negative_slope, inplace=True))
        elif t == "2":
            L.append(nn.PixelShuffle(2))
        elif t == "A":
            L.append(nn.PReLU(num_parameters=out_channels))
        else:
            raise NotImplementedError(f"Undefined type: {t}")
    return nn.Sequential(*L)


class ResBlock(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, kernel_size=3, stride=1,
                 padding=1, bias=True, mode="CRC", negative_slope=0.2):
        super().__init__()
        assert in_channels == out_channels, "ResBlock requires in==out channels"
        self.res = conv(in_channels, out_channels, kernel_size, stride, padding,
                        bias, mode, negative_slope)

    def forward(self, x):
        return x + self.res(x)


class UNetRes(nn.Module):
    """
    UNetRes (DRUNet) — grayscale denoiser.

    Parameters
    ----------
    in_nc  : int   input channels  (2 for gray: image + noise-level map)
    out_nc : int   output channels (1 for gray)
    nc     : list  channels per level, e.g. [64, 128, 256, 512]
    nb     : int   residual blocks per level
    """

    def __init__(self, in_nc=2, out_nc=1, nc=(64, 128, 256, 512), nb=4,
                 act_mode="R", downsample_mode="strideconv",
                 upsample_mode="convtranspose"):
        super().__init__()
        self.m_head = conv(in_nc, nc[0], mode="C" + act_mode)

        # ── encoder ──
        self.m_down1 = nn.Sequential(
            *[ResBlock(nc[0], nc[0], mode="C" + act_mode + "C") for _ in range(nb)],
            self._down(nc[0], nc[1], downsample_mode, act_mode))
        self.m_down2 = nn.Sequential(
            *[ResBlock(nc[1], nc[1], mode="C" + act_mode + "C") for _ in range(nb)],
            self._down(nc[1], nc[2], downsample_mode, act_mode))
        self.m_down3 = nn.Sequential(
            *[ResBlock(nc[2], nc[2], mode="C" + act_mode + "C") for _ in range(nb)],
            self._down(nc[2], nc[3], downsample_mode, act_mode))

        # ── bottleneck ──
        self.m_body = nn.Sequential(
            *[ResBlock(nc[3], nc[3], mode="C" + act_mode + "C") for _ in range(nb)])

        # ── decoder ──
        self.m_up3 = nn.Sequential(
            self._up(nc[3], nc[2], upsample_mode, act_mode),
            *[ResBlock(nc[2], nc[2], mode="C" + act_mode + "C") for _ in range(nb)])
        self.m_up2 = nn.Sequential(
            self._up(nc[2], nc[1], upsample_mode, act_mode),
            *[ResBlock(nc[1], nc[1], mode="C" + act_mode + "C") for _ in range(nb)])
        self.m_up1 = nn.Sequential(
            self._up(nc[1], nc[0], upsample_mode, act_mode),
            *[ResBlock(nc[0], nc[0], mode="C" + act_mode + "C") for _ in range(nb)])

        self.m_tail = conv(nc[0], out_nc, mode="C")

    @staticmethod
    def _down(in_nc, out_nc, mode, act):
        if mode == "strideconv":
            return conv(in_nc, out_nc, stride=2, mode="C" + act)
        elif mode == "maxpool":
            return nn.Sequential(nn.MaxPool2d(2, 2),
                                 conv(in_nc, out_nc, mode="C" + act))
        elif mode == "avgpool":
            return nn.Sequential(nn.AvgPool2d(2, 2),
                                 conv(in_nc, out_nc, mode="C" + act))
        raise ValueError(mode)

    @staticmethod
    def _up(in_nc, out_nc, mode, act):
        if mode == "convtranspose":
            return conv(in_nc, out_nc, kernel_size=2, stride=2,
                        padding=0, mode="T" + act)
        elif mode == "pixelshuffle":
            return conv(in_nc, out_nc * 4, mode="C" + act + "2")
        elif mode == "bilinear":
            return nn.Sequential(nn.Upsample(scale_factor=2, mode="bilinear",
                                             align_corners=False),
                                 conv(in_nc, out_nc, mode="C" + act))
        raise ValueError(mode)

    def forward(self, x0):
        x1 = self.m_head(x0)
        x2 = self.m_down1(x1)
        x3 = self.m_down2(x2)
        x4 = self.m_down3(x3)
        x  = self.m_body(x4)
        x  = self.m_up3(x + x4)
        x  = self.m_up2(x + x3)
        x  = self.m_up1(x + x2)
        x  = self.m_tail(x + x1)
        return x


# ── public helper ─────────────────────────────────────────────────────────────

def load_drunet(model_path: str, device: torch.device) -> UNetRes:
    """Load pretrained DRUNet weights from *model_path* (drunet_gray.pth)."""
    model = UNetRes(in_nc=2, out_nc=1, nc=[64, 128, 256, 512], nb=4,
                    act_mode="R", downsample_mode="strideconv",
                    upsample_mode="convtranspose")
    model.load_state_dict(torch.load(model_path, map_location=device), strict=True)
    model.eval()
    return model.to(device)


def drunet_denoise(model: UNetRes, x: torch.Tensor, sigma: float,
                   device: torch.device, refield: int = 32) -> torch.Tensor:
    """
    Apply DRUNet to tensor *x* (1×1×H×W, values in [0,1]).
    Pads to a multiple of *refield* if needed.
    Returns denoised tensor of the same shape.
    """
    _, _, H, W = x.shape
    pad_h = (refield - H % refield) % refield
    pad_w = (refield - W % refield) % refield
    if pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

    noise_map = torch.full_like(x, sigma / 255.0)
    inp = torch.cat([x, noise_map], dim=1).to(device)
    with torch.no_grad():
        out = model(inp)
    out = out[:, :, :H, :W]
    return out.clamp(0.0, 1.0)
