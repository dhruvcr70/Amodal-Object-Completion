"""Compact RGB U-Net for replacing occluded regions in product images."""
import torch
from torch import nn
from torch.nn import functional as F


def _groups(channels: int) -> int:
    for count in (8, 4, 2):
        if channels % count == 0:
            return count
    return 1


class DoubleConv(nn.Sequential):
    def __init__(self, cin: int, cout: int):
        super().__init__(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(cout), cout), nn.SiLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(cout), cout), nn.SiLU(inplace=True),
        )


class RGBCompletionUNet(nn.Module):
    """Map RGB plus visible-object mask to an RGB estimate of the full image."""
    def __init__(self, base: int = 16):
        super().__init__()
        self.enc1 = DoubleConv(4, base)
        self.enc2 = DoubleConv(base, base * 2)
        self.enc3 = DoubleConv(base * 2, base * 4)
        self.center = DoubleConv(base * 4, base * 8)
        self.dec3 = DoubleConv(base * 12, base * 4)
        self.dec2 = DoubleConv(base * 6, base * 2)
        self.dec1 = DoubleConv(base * 3, base)
        self.output = nn.Conv2d(base, 3, 1)

    @staticmethod
    def up(x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        center = self.center(F.max_pool2d(e3, 2))
        d3 = self.dec3(torch.cat((self.up(center, e3), e3), dim=1))
        d2 = self.dec2(torch.cat((self.up(d3, e2), e2), dim=1))
        d1 = self.dec1(torch.cat((self.up(d2, e1), e1), dim=1))
        return torch.tanh(self.output(d1))


def composite_known_pixels(prediction: torch.Tensor, observed: torch.Tensor,
                           occluder_mask: torch.Tensor) -> torch.Tensor:
    """Keep the observed scene exact and use the model only under the blocker."""
    return torch.where(occluder_mask.bool().expand_as(observed), prediction, observed)
