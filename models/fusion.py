from torch import nn
from typing import List, Tuple
import torch

class ConvFuser(nn.Sequential):

    def __init__(self, in_channels: int, out_channels: int) -> None:
        self.in_channels = in_channels
        self.out_channels = out_channels
        super().__init__(
            nn.Conv2d(
                sum(in_channels), out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        return super().forward(torch.cat(inputs, dim=1))

class FusionLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.fuser = ConvFuser(in_channels=[80, 256], out_channels=256)