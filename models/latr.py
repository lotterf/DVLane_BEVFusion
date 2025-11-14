import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import *
# from mmdet.core import multi_apply
# from mmdet.models.builder import build_loss
# from mmcv.utils import Config
# from .backbone import ImageBackbone, PointCloudBackbone, SECOND_Module
# from .neck import ImageNeck, ViewTransform
from mmdet.models.builder import build_backbone, build_neck
from mmdet3d.models.builder import build_backbone as build_3d_backbone
from .dv3dlane_head import DV3DLaneHead
from .bevfusion.ops import Voxelization

# from PIL import Image
# import matplotlib.pyplot as plt

class LATR(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.no_cuda = args.no_cuda
        self.batch_size = args.batch_size
        self.num_lane_type = 1  # no centerline
        self.num_y_steps = args.num_y_steps
        self.max_lanes = args.max_lanes
        self.num_category = args.num_category
        self.num_pt_per_line = args.num_pt_per_line
        _dim_ = args.latr_cfg.fpn_dim
        num_query = args.latr_cfg.num_query
        num_group = args.latr_cfg.num_group
        self.num_query = num_query
        pts_cfg = args.pts_module_cfg

        self.encoder = build_backbone(args.latr_cfg.encoder)
        if hasattr(self.encoder, 'init_weights'):
            self.encoder.init_weights()
        self.neck = build_neck(args.latr_cfg.neck)
        self.view_transform = build_neck(args.latr_cfg.view_transform)
        
        voxel_layer_cfg = pts_cfg.pts_voxel_layer.copy()
        self.voxelize_reduce = voxel_layer_cfg.pop('voxelize_reduce')
        voxel_layer_cfg.pop('type')
        self.pts_voxel_layer = Voxelization(**voxel_layer_cfg)

        self.pts_encoder = build_3d_backbone(pts_cfg.pts_voxel_encoder)
        self.pts_bev_backbone = build_backbone(pts_cfg.pts_bev_backbone)
        self.pts_bev_neck = build_neck(pts_cfg.pts_bev_neck)
        self.fusion_layer = build_neck(args.latr_cfg.fusion_layer)
        self.reduce = nn.Conv2d(512, 256, 1)

        head_extra_cfgs = args.latr_cfg.get('head', {})
        assert head_extra_cfgs.get('pred_dim', self.num_y_steps) == self.num_y_steps
        head_extra_cfgs['pred_dim'] = self.num_y_steps
        # build 2d query-based instance seg
        self.head = DV3DLaneHead(
            args=args,
            dim=_dim_,
            num_group=num_group,
            num_convs=4,  #每个分支/头部的卷积层数
            in_channels=_dim_,
            kernel_dim=_dim_,
            position_range=args.position_range,  #BEV坐标的起始范围（通常是最小值），用于把真实坐标归一化到网格、做位置编码或反归一化
            pos_encoding_2d=args.latr_cfg.pos_encoding_2d, #对图像特征做的 2D 位置编码配置
            q_pos_emb=args.latr_cfg.q_pos_emb, #query 的位置/先验嵌入配置
            pos_encoding_bev=args.latr_cfg.pos_encoding_bev, #BEV 平面的位置编码配置
            num_query=num_query,
            num_classes=args.num_category,
            embed_dims=_dim_,
            transformer=args.transformer,
            sparse_ins_decoder=args.sparse_ins_decoder,
            **head_extra_cfgs,
            trans_params=args.latr_cfg.get('trans_params', {})
        )

    def voxelize(self, points):
        feats, coords, sizes = [], [], []
        for k, res in enumerate(points):
            ret = self.pts_voxel_layer(res)
            f, c = ret if len(ret) == 2 else ret[:2]
            n = ret[2] if len(ret) == 3 else None
            feats.append(f)
            coords.append(F.pad(c, (1, 0), mode='constant', value=k))
            if n is not None:
                sizes.append(n)
        feats = torch.cat(feats, dim=0)
        coords = torch.cat(coords, dim=0)
        if sizes:
            sizes = torch.cat(sizes, dim=0)
            if self.voxelize_reduce:
                feats = feats.sum(dim=1) / sizes.type_as(feats).view(-1, 1)
        return feats, coords, sizes

    def forward(self, image, point_cloud, _M_inv=None, is_training=True, extra_dict=None):

        # 图像 BEV
        out_featList = self.encoder(image)
        neck_out = self.neck(out_featList)
        neck_out = neck_out[0] #[2,256,45,60]
        img_bev_out = self.view_transform(
            neck_out,
            point_cloud,
            extra_dict['lidar2img'],
            extra_dict['intrinsics'],
            extra_dict['cam2lidar']
        ).permute(0,1,3,2) #[2,80,250,300]

        # 点云 BEV
        points = [p.squeeze(0) for p in point_cloud]
        feats, coords, sizes = self.voxelize(points) #[12000,4] [12000,3] [12000]
        point_bev_out = self.pts_encoder(feats, coords, coords[-1, 0] + 1).permute(0,1,3,2)  #[2,256,250,300]

        fusion_features = self.fusion_layer([img_bev_out, point_bev_out]) #[2,256,250,300]
        fusion_features = self.pts_bev_backbone(fusion_features)
        fusion_features = self.pts_bev_neck(fusion_features)
        fusion_features = self.reduce(fusion_features[0])

        extra_dict['x'] = [fusion_features]
        output = self.head(extra_dict, is_training=is_training)
        return output
