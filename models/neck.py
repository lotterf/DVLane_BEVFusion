from torch import nn
from .bevfusion_necks import GeneralizedLSSFPN
from .bevfusion.depth_lss_fusion import DepthLSSTransform

class ImageNeck(nn.Module):
    def __init__(self):
        super().__init__()
        self.neck = GeneralizedLSSFPN(
            in_channels=[192, 384, 768], out_channels=256, start_level=0, num_outs=3,
            norm_cfg=dict(type='BN2d', requires_grad=True), act_cfg=dict(type='ReLU', inplace=True),
            upsample_cfg=dict(mode='bilinear', align_corners=False)
        )

class ViewTransform(nn.Module):
    def __init__(self):
        super().__init__()
        self.transform = DepthLSSTransform(
            in_channels=256, out_channels=80, image_size=[360, 480], feature_size=[45, 60],
            xbound=[-30.0, 30.0, 0.1], ybound=[3.0, 103.0, 0.2], zbound=[-3.0, 6.0, 9.0],
            dbound=[3.0, 103.0, 0.5], downsample=2
        )