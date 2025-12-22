from torch import nn
from typing import List, Tuple
import torch
import random
from mmdet.models.builder import NECKS 

@NECKS.register_module() 
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
    
@NECKS.register_module()
class GlobalAlign(nn.Module):
    def __init__(self, in_channels: list, out_channels: int) -> None:
        super().__init__()
        self.img_channel = in_channels[0]
        self.lidar_channel = in_channels[1]
        self.out_channel = out_channels

        self.conv = nn.Sequential(
            nn.Conv2d(sum(in_channels), out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )

        self.offset_conv = nn.Conv2d(sum(in_channels), 2, kernel_size=3, stride=1, padding=1)
        
        self.deform_conv = nn.Conv2d(self.lidar_channel, self.lidar_channel, kernel_size=3, stride=1, padding=1)

    def forward(self, inputs: list) -> torch.Tensor:
        img_bev = inputs[0]
        lidar_bev = inputs[1]

        cat_bev = torch.cat([img_bev, lidar_bev], dim=1)
        mm_bev = self.conv(cat_bev) 

        if self.training:
            shift_x = random.randint(0, 5)
            shift_y = random.randint(0, 5)
        else:
            shift_x = 0
            shift_y = 0

        shifted_img_bev = torch.roll(img_bev, shifts=(shift_x, shift_y), dims=(3, 2))
        
        offset = self.offset_conv(torch.cat([shifted_img_bev, lidar_bev], dim=1))
        offset = offset.permute(0, 2, 3, 1)

        return mm_bev
