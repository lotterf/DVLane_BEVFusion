from torch import nn
from mmdet.models.backbones.swin import SwinTransformer
from .bevfusion.sparse_encoder import BEVFusionSparseEncoder
from .bevfusion.ops import Voxelization
from .second_module import SECOND, SECONDFPN

class ImageBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.swin = SwinTransformer(
            embed_dims=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24], window_size=7,
            mlp_ratio=4, qkv_bias=True, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.2,
            patch_norm=True, out_indices=[1, 2, 3], with_cp=False, convert_weights=True,
            init_cfg=dict(type='Pretrained', checkpoint='https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth')
        )
    def init_weights(self):
        self.swin.init_weights()

class PointCloudBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        voxelize_cfg = dict(
            max_num_points=15, point_cloud_range=[-30.0, 3.0, -3.0, 30.0, 103.0, 6.0],
            voxel_size=[0.025, 0.05, 0.225], max_voxels=[6000, 8000], voxelize_reduce=True
        )
        self.voxelize_reduce = voxelize_cfg.pop('voxelize_reduce')
        self.pts_voxel_layer = Voxelization(**voxelize_cfg)
        self.encoder = BEVFusionSparseEncoder(
            in_channels=3, sparse_shape=[2400, 2000, 41], order=('conv', 'norm', 'act'),
            norm_cfg=dict(type='BN1d', eps=0.001, momentum=0.01),
            encoder_channels=((16, 16, 32), (32, 32, 64), (64, 64, 128), (128, 128)),
            encoder_paddings=((0, 0, 1), (0, 0, 1), (0, 0, (1, 1, 0)), (0, 0)),
            block_type='basicblock'
        )

class SECOND_Module(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = SECOND(
        in_channels=256,
        out_channels=[128, 256],
        layer_nums=[5, 5],
        layer_strides=[1, 2],
        norm_cfg=dict(type='BN', eps=0.001, momentum=0.01),
        conv_cfg=dict(type='Conv2d', bias=False))
        self.neck = SECONDFPN(
        in_channels=[128, 256],
        out_channels=[256, 256],
        upsample_strides=[1, 2],
        norm_cfg=dict(type='BN', eps=0.001, momentum=0.01),
        upsample_cfg=dict(type='deconv', bias=False),
        use_conv_for_no_stride=True)