from torch import nn
from mmdet.models.backbones.swin import SwinTransformer
from .bevfusion.sparse_encoder import BEVFusionSparseEncoder
from .bevfusion.ops import Voxelization
from .second_module import SECOND, SECONDFPN
from mmdet.models.builder import build_backbone as build_2d_backbone, build_neck as build_2d_neck
from mmdet3d.models.builder import build_backbone as build_3d_backbone

class ImageBackbone(nn.Module):
    # 接受配置字典 cfg 作为参数
    def __init__(self, cfg):
        super().__init__()
        # 使用 build_backbone 动态实例化，cfg 即为 fusionlane_300_baseline_lite.py 中的 encoder 配置
        # cfg = dict(type='SwinTransformer', embed_dims=96, ...)
        self.swin = build_2d_backbone(cfg)
        
    def init_weights(self):
        # 调用通过配置实例化的模型的 init_weights 方法
        if hasattr(self.swin, 'init_weights'):
             self.swin.init_weights()

class PointCloudBackbone(nn.Module):
    # 修改为接收配置字典
    def __init__(self, cfg):
        super().__init__()
        # 使用配置构建 Voxelization (pts_voxel_layer)
        self.voxelize_reduce = cfg.voxelize_cfg.pop('voxelize_reduce')
        self.pts_voxel_layer = Voxelization(**cfg.voxelize_cfg)
        # 使用配置构建 Sparse Encoder (encoder)
        self.encoder = build_3d_backbone(cfg.sparse_encoder_cfg)

class SECOND_Module(nn.Module):
    # 修改为接收配置字典
    def __init__(self, cfg):
        super().__init__()
        # 使用配置构建 SECOND Backbone
        self.backbone = build_2d_backbone(cfg.second_backbone_cfg)
        # 使用配置构建 SECOND FPN (Neck)
        self.neck = build_2d_neck(cfg.second_neck_cfg)
