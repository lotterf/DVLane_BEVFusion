from torch import nn
from .bevfusion_necks import GeneralizedLSSFPN
from .bevfusion.depth_lss_fusion import DepthLSSTransform
from mmdet.models.builder import build_neck

class ImageNeck(nn.Module):
    # 接受配置字典 cfg 作为参数
    def __init__(self, cfg):
        super().__init__()
        # 使用 build_neck 动态实例化，cfg 即为 fusionlane_300_baseline_lite.py 中的 neck 配置
        self.neck = build_neck(cfg)

class ViewTransform(nn.Module):
    # 修改为接收配置字典
    def __init__(self, cfg):
        super().__init__()
        # 使用配置构建
        self.transform = build_neck(cfg) # DepthLSSTransform 应注册为 Neck