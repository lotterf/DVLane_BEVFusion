import numpy as np
import math
import cv2
import warnings
from functools import partial

import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F

from fvcore.nn.weight_init import c2_msra_fill, c2_xavier_fill

from mmdet.models.utils.builder import TRANSFORMER
from mmcv.cnn import (build_activation_layer, build_conv_layer,
                    build_norm_layer, xavier_init, constant_init)
from mmcv.runner.base_module import BaseModule
from mmcv.cnn.bricks.transformer import (BaseTransformerLayer,
                                        TransformerLayerSequence,
                                        build_transformer_layer_sequence)
from mmcv.cnn.bricks.registry import (ATTENTION,TRANSFORMER_LAYER,
                                    TRANSFORMER_LAYER_SEQUENCE)
from mmcv.ops.multi_scale_deform_attn import MultiScaleDeformableAttention
from mmcv.cnn.bricks.transformer import build_attention

from .utils import inverse_sigmoid
from .lidar_module_utils import *

def generate_ref_pt(minx, miny, maxx, maxy, z, nx, ny, device='cuda'):
    if isinstance(z, list):
        nz = z[-1]
        # minx, miny, maxx, maxy : in ground coords
        xs = torch.linspace(minx, maxx, nx, dtype=torch.float, device=device
                ).view(1, -1, 1).expand(ny, nx, nz)
        ys = torch.linspace(miny, maxy, ny, dtype=torch.float, device=device
                ).view(-1, 1, 1).expand(ny, nx, nz)
        zs = torch.linspace(z[0], z[1], nz, dtype=torch.float, device=device
                ).view(1, 1, -1).expand(ny, nx, nz)
        ref_3d = torch.stack([xs, ys, zs], dim=-1)
        ref_3d = ref_3d.flatten(1, 2)
    else:
        # minx, miny, maxx, maxy : in ground coords
        xs = torch.linspace(minx, maxx, nx, dtype=torch.float, device=device
                ).view(1, -1, 1).expand(ny, nx, 1)
        ys = torch.linspace(miny, maxy, ny, dtype=torch.float, device=device
                ).view(-1, 1, 1).expand(ny, nx, 1)
        ref_3d = F.pad(torch.cat([xs, ys], dim=-1), (0, 1), mode='constant', value=z)
    return ref_3d


@TRANSFORMER_LAYER.register_module()
class DV3DLaneDecoderLayer(BaseTransformerLayer):
    def __init__(self,
                 attn_cfgs,
                 feedforward_channels,
                 ffn_dropout=0.0,
                 operation_order=None,
                 act_cfg=dict(type='ReLU', inplace=True),
                 norm_cfg=dict(type='LN'),
                 ffn_num_fcs=2,
                 pv_attn_cfg=None, # [新增] PV Attention 配置
                 **kwargs):
        super().__init__(
            attn_cfgs=attn_cfgs,
            feedforward_channels=feedforward_channels,
            ffn_dropout=ffn_dropout,
            operation_order=operation_order,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            ffn_num_fcs=ffn_num_fcs,
            **kwargs)
        assert len(operation_order) == 6
        assert set(operation_order) == set(
            ['self_attn', 'norm', 'cross_attn', 'ffn'])
        
        # [新增] 初始化 PV Cross Attention
        self.pv_attn_cfg = pv_attn_cfg
        if pv_attn_cfg is not None:
            self.cross_attn_pv = build_attention(pv_attn_cfg)
            self.norm_pv = build_norm_layer(norm_cfg, self.embed_dims)[1]

    def forward(self,
                query,
                key=None,
                value=None,
                query_pos=None,
                key_pos=None,
                attn_masks=None,
                query_key_padding_mask=None,
                key_padding_mask=None,
                pv_feats=None,             # [新增]
                pv_reference_points=None,  # [新增]
                pv_spatial_shapes=None,    # [新增]
                pv_level_start_index=None, # [新增]
                **kwargs):
        
        # 1. 原始 BEV 流程 (Self Attn -> Cross Attn BEV -> FFN)
        # kwargs 中包含了 BEV 需要的 reference_points, spatial_shapes 等
        query = super().forward(
            query=query, key=key, value=value,
            query_pos=query_pos, key_pos=key_pos,
            attn_masks=attn_masks, query_key_padding_mask=query_key_padding_mask,
            key_padding_mask=key_padding_mask, **kwargs)
        
        # 2. [新增] PV Cross-View Interaction
        if self.pv_attn_cfg is not None and pv_feats is not None:
            identity = query
            if pv_reference_points is not None:
                # [修复] 创建专用的 kwargs，移除与 BEV 相关的参数，避免冲突
                pv_kwargs = kwargs.copy()
                keys_to_pop = ['reference_points', 'spatial_shapes', 'level_start_index']
                for k in keys_to_pop:
                    if k in pv_kwargs:
                        pv_kwargs.pop(k)
                
                query = self.cross_attn_pv(
                    query=query,
                    key=None, # Deformable Attn 不需要 Key
                    value=pv_feats,
                    identity=identity,
                    query_pos=query_pos,
                    reference_points=pv_reference_points, # 投影后的 2D 参考点
                    spatial_shapes=pv_spatial_shapes,
                    level_start_index=pv_level_start_index,
                    **pv_kwargs 
                )
                query = self.norm_pv(query)
            
        return query


@TRANSFORMER_LAYER_SEQUENCE.register_module()
class DV3DLaneTransformerDecoder(TransformerLayerSequence):
    def __init__(self,
                 *args,
                 embed_dims=None,
                 post_norm_cfg=None, # dict(type='LN'),
                 M_decay_ratio=10,
                 num_query=None,
                 num_anchor_per_query=None,
                 anchor_y_steps=None,
                 num_lidar_feat=3,
                 look_forward_twice=False,
                 return_intermediate=True,
                #  use_vismask=False,
                #  use_refpt_qpos=False,
                 refpt_qpos_encode='mlp',
                 rept_pe_cfg=dict(
                    type='mlp',
                    inc=3,
                    hidc=256*4,
                    outc=256,
                 ),
                 refpt_pe_dim='3d',
                 pc_range=None,
                 mamba_attention_cfg=None,
                 uniseq_pos_embed=False,
                 max_seq_len=20,
                 encode_last_refpt_pos_in_query=False,
                 **kwargs):
        super(DV3DLaneTransformerDecoder, self).__init__(*args, **kwargs)
        assert num_lidar_feat >= 3
        self.num_lidar_feat = num_lidar_feat
        self.look_forward_twice = look_forward_twice
        self.return_intermediate = return_intermediate

        if post_norm_cfg is not None:
            self.post_norm = build_norm_layer(post_norm_cfg,
                                              self.embed_dims)[1]
        else:
            self.post_norm = None

        self.num_query = num_query
        self.num_anchor_per_query = num_anchor_per_query
        self.anchor_y_steps = anchor_y_steps
        self.num_points_per_anchor = len(anchor_y_steps) // num_anchor_per_query

        self.embed_dims = embed_dims
        self.refpt_qpos_encode = refpt_qpos_encode
        self.refpt_pe_dim = refpt_pe_dim
        self.pc_range = pc_range

    def init_weights(self):
        super().init_weights()

    def forward(self, query, key, value,
                top_view_region=None,
                bev_h=None, bev_w=None,
                init_z=0, img_feats=None,
                lidar2img=None, pad_shape=None,
                key_pos=None, key_padding_mask=None,
                sin_embed=None, reference_points=None,
                reg_branches=None, cls_branches=None,
                query_pos=None, points=None,
                pv_feats=None, pv_spatial_shapes=None, pv_level_start_index=None, # [新增]
                **kwargs):
        assert key_padding_mask is None

        # init pts and M to generate pos embed for key/value
        xmin = top_view_region[0]
        ymin = top_view_region[1]
        zmin = top_view_region[2]
        xmax = top_view_region[3]
        ymax = top_view_region[4]
        zmax = top_view_region[5]

        intermediate = []
        project_results = []
        outputs_classes = []
        outputs_coords = []

        last_reference_points = []

        if key_pos is not None:
            sin_embed = key_pos + sin_embed
        sin_embed = sin_embed.permute(1, 0, 2).contiguous() #torch.Size([75000, 1, 256])

        B = key.shape[1]
        last_reference_points = [reference_points]

        for layer_idx, layer in enumerate(self.layers):
            
            # [新增] 投影逻辑: 3D Query Points -> 2D Image Plane
            pv_reference_points = None
            if pv_feats is not None and lidar2img is not None:
                # [修复] 处理 reference_points 形状不一致问题
                if reference_points.dim() == 3:
                    # Initial state: [BS, Num_Query, Num_Anchor * Num_Pts * 2] -> Reshape to 5D for calculation
                    ref_pts_reshaped = reference_points.view(
                        B, self.num_query, self.num_anchor_per_query,
                        self.num_points_per_anchor, 2
                    )
                else:
                    # Already reshaped in previous iteration
                    ref_pts_reshaped = reference_points

                # 开始计算投影
                # ref_pts_reshaped: [BS, N_Query, N_Anchor, N_Pts, 2]
                bs, n_query, n_anchor, n_pts, _ = ref_pts_reshaped.shape
                
                # Flatten 用于批量投影: [BS, N*A*P, 2]
                ref_pts_flat = ref_pts_reshaped.view(bs, -1, 2) 
                
                # 1. 恢复到真实世界坐标
                x_real = ref_pts_flat[..., 0] * (xmax - xmin) + xmin
                y_real = ref_pts_flat[..., 1] * (ymax - ymin) + ymin
                z_real = torch.full_like(x_real, init_z) 
                
                points_3d = torch.stack([x_real, y_real, z_real], dim=-1) # [BS, M, 3]
                ones = torch.ones_like(x_real).unsqueeze(-1)
                points_homo = torch.cat([points_3d, ones], dim=-1) # [BS, M, 4]
                
                # 2. 投影到图像平面 (lidar2img @ points)
                points_cam = torch.bmm(lidar2img, points_homo.permute(0, 2, 1)).permute(0, 2, 1) # [BS, M, 4]
                
                z_cam = torch.clamp(points_cam[..., 2:3], min=1e-5)
                u = points_cam[..., 0:1] / z_cam
                v = points_cam[..., 1:2] / z_cam
                
                # 3. 归一化到 [0, 1]
                # [修复] 鲁棒地解析 pad_shape，兼容 Tensor (B, 2)、List[Tensor]、List[List] 等情况
                img_shape = None
                
                if isinstance(pad_shape, list):
                    img_shape = pad_shape[0]
                elif isinstance(pad_shape, torch.Tensor):
                     if pad_shape.dim() == 2: # (Batch, 3) or (Batch, 2)
                         img_shape = pad_shape[0]
                     else:
                         img_shape = pad_shape

                if isinstance(img_shape, torch.Tensor):
                    if img_shape.dim() == 2 and img_shape.shape[0] == 1:
                        H_img = img_shape[0][0]
                        W_img = img_shape[0][1]
                    elif img_shape.dim() == 1 and img_shape.shape[0] >= 2:
                        H_img = img_shape[0]
                        W_img = img_shape[1]
                    else:
                        H_img = img_shape[0]
                        W_img = img_shape[1] if img_shape.numel() > 1 else H_img
                else: # Tuple or List
                    H_img = img_shape[0]
                    W_img = img_shape[1]
                
                u_norm = u / W_img
                v_norm = v / H_img
                
                pv_ref_raw = torch.cat([u_norm, v_norm], dim=-1) # [BS, M, 2]
                
                # 4. [关键修复] 不要取均值，直接保留 M=800 个点
                # pv_ref_raw shape: [BS, M, 2]
                
                num_pv_levels = len(pv_spatial_shapes) if pv_spatial_shapes is not None else 1
                
                # 直接增加 Level 维度并扩展: [BS, M, 2] -> [BS, M, 1, 2] -> [BS, M, Levels, 2]
                pv_reference_points = pv_ref_raw.unsqueeze(2).expand(-1, -1, num_pv_levels, -1)
                
                # 确保在 [0, 1] 范围内
                pv_reference_points = torch.clamp(pv_reference_points, 0, 1)

            query = layer(query, key=key, value=value,
                          key_pos=sin_embed,
                          reference_points=reference_points, # 传给 Layer 原始形状
                          pc_range=[xmin, ymin, zmin, xmax, ymax, zmax],
                          pad_shape=pad_shape,
                          lidar2img=lidar2img,
                          query_pos=query_pos,
                          layer_idx=layer_idx,
                          pv_feats=pv_feats,             # [传递]
                          pv_reference_points=pv_reference_points, # [传递计算好的 2D 参考点]
                          pv_spatial_shapes=pv_spatial_shapes,     # [传递]
                          pv_level_start_index=pv_level_start_index, # [传递]
                          **kwargs)

            if self.return_intermediate:
                if self.post_norm is not None:
                    intermediate.append(self.post_norm(query))
                else:
                    intermediate.append(query)
            query = query.permute(1, 0, 2).contiguous()
            tmp = reg_branches[layer_idx](query)

            bs = tmp.shape[0]
            # iterative update
            tmp = tmp.view(bs, self.num_query,
                self.num_anchor_per_query, -1, 3)

            reference_points = reference_points.view(
                bs, self.num_query, self.num_anchor_per_query,
                self.num_points_per_anchor, 2
            )
            reference_points = inverse_sigmoid(reference_points)
            new_reference_points = torch.stack([
                reference_points[..., 0] + tmp[..., 0],
                reference_points[..., 1] + tmp[..., 1],
            ], dim=-1)
            new_reference_points = new_reference_points.sigmoid()

            # detrex DINO vs deform-detr
            reference_points = new_reference_points.detach()
            last_reference_not_detach = inverse_sigmoid(
                last_reference_points[-1]).view(
                    bs, self.num_query, self.num_anchor_per_query,
                    self.num_points_per_anchor, 2
                )
            lftwice_refpts = torch.stack([
                last_reference_not_detach[..., 0] + tmp[..., 0],
                last_reference_not_detach[..., 1] + tmp[..., 1],
            ], dim=-1).sigmoid()

            outputs_coords.append(
                torch.cat([
                    lftwice_refpts,
                    tmp[..., -1:]], dim=-1))
            last_reference_points.append(new_reference_points)

            cls_feat = query.view(
                bs, self.num_query, self.num_anchor_per_query, -1)
            cls_feat = torch.max(cls_feat, dim=2)[0]
            outputs_class = cls_branches[layer_idx](cls_feat)

            outputs_classes.append(outputs_class)
            query = query.permute(1, 0, 2).contiguous()

        if self.return_intermediate:
            return torch.stack(intermediate).permute(0, 2, 1, 3).contiguous(), project_results, outputs_classes, outputs_coords
        else:
            return query, project_results, outputs_classes, outputs_coords


@TRANSFORMER.register_module()
class DV3DLaneTransformer(BaseModule):
    def __init__(self, encoder=None, decoder=None, init_cfg=None):
        super(DV3DLaneTransformer, self).__init__(init_cfg=init_cfg)
        if encoder is not None:
            self.encoder = build_transformer_layer_sequence(encoder)
        else:
            self.encoder = None
        self.decoder = build_transformer_layer_sequence(decoder)
        self.embed_dims = self.decoder.embed_dims
        self.init_weights()

    def init_weights(self):
        # follow the official DETR to init parameters
        for m in self.modules():
            if hasattr(m, 'weight') and m.weight.dim() > 1:
                xavier_init(m, distribution='uniform')
        self._is_init = True

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device): #获取 reference points
        reference_points_list = []
        for lvl, (H, W) in enumerate(spatial_shapes):
            #  TODO  check this 0.5
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(
                    0.5, H - 0.5, H, dtype=torch.float32, device=device),
                torch.linspace(
                    0.5, W - 0.5, W, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def get_valid_ratio(self, mask): #获取有效区域比例
        """Get the valid radios of feature maps of all  level."""
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    @property
    def with_encoder(self):
        return hasattr(self, 'encoder') and self.encoder

    def forward(self, x, mask, query,
                query_embed, pos_embed,
                reference_points=None,
                reg_branches=None, cls_branches=None,
                spatial_shapes=None,
                level_start_index=None,
                mlvl_masks=None,
                mlvl_positional_encodings=None,
                pos_embed2d=None,
                key_pos=None,
                pv_feats=None, pv_spatial_shapes=None, pv_level_start_index=None, # [新增]
                **kwargs):
        # assert pos_embed is None
        memory = x
        # encoder
        if hasattr(self, 'encoder') and self.encoder:
            B = x.shape[1]
            # mlvl_masks = [torch.zeros((B, *s),
            #                          dtype=torch.bool, device=x.device)
            #     for s in spatial_shapes]
            valid_ratios = torch.stack(
                [self.get_valid_ratio(m) for m in mlvl_masks], 1)
            reference_points_2d = \
                self.get_reference_points(spatial_shapes,
                                          valid_ratios,
                                          device=x.device)
            memory = self.encoder(
                query=memory,
                key=memory,
                value=memory,
                key_pos=key_pos,
                query_pos=pos_embed2d,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                reference_points=reference_points_2d,
                valid_ratios=valid_ratios,
            )

        if query_embed is not None:
            query_embed = query_embed.permute(1, 0, 2).contiguous() #torch.Size([800, 1, 256])
        if mask is not None:
            mask = mask.view(bs, -1)
        if query is not None:
            query = query.permute(1, 0, 2).contiguous() #torch.Size([800, 1, 256])

        out_dec, project_results, outputs_classes, outputs_coords = \
            self.decoder(
                query=query,
                key=memory,
                value=memory,
                key_pos=pos_embed,
                query_pos=query_embed,
                key_padding_mask=mask.astype(torch.bool) if mask is not None else None,
                reg_branches=reg_branches,
                cls_branches=cls_branches,
                reference_points=reference_points,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                pv_feats=pv_feats,                         # [新增]
                pv_spatial_shapes=pv_spatial_shapes,       # [新增]
                pv_level_start_index=pv_level_start_index, # [新增]
                **kwargs
            )
        return out_dec, project_results, \
               outputs_classes, outputs_coords
