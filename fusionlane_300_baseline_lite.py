import numpy as np
from mmcv.utils import Config

_base_ = [
    '../_base_/base_res101_bs16xep100.py',  # 可能需要调整为你的基础配置
    '../_base_/optimizer.py'
]

nepochs = 24
eval_freq = 1

optimizer_cfg = dict(
    type='AdamW',
    lr=2e-4,
    betas=(0.95, 0.99),
    paramwise_cfg=dict(
        custom_keys={
            'sampling_offsets': dict(lr_mult=0.1),
        }),
    weight_decay=0.01)

clip_grad_norm = 20

dataset = '300'
dataset_dir = './data/openlane/data_final/'
data_dir = './data/openlane/lane3d_300/'
vis_dir = '/media/home/data_share/OpenLane/waymo/data/results/data_final/vis_results/'
save_pred = False

batch_size = 1
nworkers = 10

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

resize_h = 360
resize_w = 480

photo_aug = dict(
    brightness_delta=32 // 2,
    contrast_range=(0.5, 1.5),
    saturation_range=(0.5, 1.5),
    hue_delta=9)

seg_bev = True
bev_thick = 8
front_thick = 8
num_lidar_feat = 6

top_view_region = np.array([
    [-10, 103], [10, 103], [-10, 3], [10, 3]]) # 定义了鸟瞰图的前后左右边界
enlarge_length_x = 20 #扩展 top view 范围
position_range = [ #[-30,3,-5,30,103,5]
    top_view_region[0][0] - enlarge_length_x,
    top_view_region[2][1],
    -5,
    top_view_region[1][0] + enlarge_length_x,
    top_view_region[0][1],
    5.]

voxel_size = (0.2, 0.4, 10)
grid_size = [
    int((position_range[i + 3] - position_range[i]) / voxel_size[i])
    for i in range(3)]

anchor_y_steps = np.linspace(3, 103, 20)
pred_dim = len(anchor_y_steps)
num_y_steps = len(anchor_y_steps)

_dim_ = 256 # Transformer 嵌入维度
num_query = 40 # 检测查询数量
num_pt_per_line = 20 # 每条车道线的点数
num_category = 21 # 类别数量 (含背景)
pos_threshold = 0.3 # F1 score 阈值

pts_module_cfg = dict(
    # Voxelization Config (pts_voxel_layer)
    pts_voxel_layer=dict(
        type='Voxelization',
        max_num_points=15, 
        point_cloud_range=[-30.0, 3.0, -3.0, 30.0, 103.0, 6.0],
        voxel_size=[0.025, 0.05, 0.225], 
        max_voxels=[6000, 8000], 
        voxelize_reduce=True
    ),
    # Sparse Encoder Config (PointCloudBackbone.encoder)
    pts_voxel_encoder=dict(
        type='BEVFusionSparseEncoder',
        in_channels=3, 
        sparse_shape=[2400, 2000, 41],
        order=('conv', 'norm', 'act'),
        norm_cfg=dict(type='BN1d', eps=0.001, momentum=0.01),
        encoder_channels=((16, 16, 32), (32, 32, 64), (64, 64, 128), (128, 128)),
        encoder_paddings=((0, 0, 1), (0, 0, 1), (0, 0, (1, 1, 0)), (0, 0)),
        block_type='basicblock'
    ),
    # SECOND Backbone (SECOND_Module.backbone)
    pts_bev_backbone=dict(
        type='SECOND',
        in_channels=256,
        out_channels=[128, 256],
        layer_nums=[5, 5],
        layer_strides=[1, 2],
        norm_cfg=dict(type='BN', eps=0.001, momentum=0.01),
        conv_cfg=dict(type='Conv2d', bias=False)
    ),
    # SECOND Neck (SECOND_Module.neck)
    pts_bev_neck=dict(
        type='SECONDFPN',
        in_channels=[128, 256],
        out_channels=[256, 256],
        upsample_strides=[1, 2],
        norm_cfg=dict(type='BN', eps=0.001, momentum=0.01),
        upsample_cfg=dict(type='deconv', bias=False),
        use_conv_for_no_stride=True
    ),
)

latr_cfg = dict(
    fpn_dim=_dim_,
    num_query=num_query,
    num_group=1,
    sparse_num_group=4,
    encoder=dict(
        type='SwinTransformer',
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=[1, 2, 3],
        with_cp=False,
        convert_weights=True,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth')
    ),
    neck=dict(
        type='GeneralizedLSSFPN',
        in_channels=[192, 384, 768],
        out_channels=_dim_,
        start_level=0,
        num_outs=3,
        norm_cfg=dict(type='BN2d', requires_grad=True),
        act_cfg=dict(type='ReLU', inplace=True),
        upsample_cfg=dict(mode='bilinear', align_corners=False)
    ),
    view_transform=dict(
        type='DepthLSSTransform',
        in_channels=_dim_,
        out_channels=80,
        image_size=[360, 480],
        feature_size=[45, 60],
        xbound=[-30.0, 30.0, 0.1],
        ybound=[3.0, 103.0, 0.2],
        zbound=[-3.0, 6.0, 9.0],
        dbound=[3.0, 103.0, 0.5],
        downsample=2
    ),

    pos_encoding_2d=dict(
        type='SinePositionalEncoding',
        num_feats=_dim_ // 2, normalize=True),
    q_pos_emb=dict(
        type='LiDAR_XYZ_PE',
        pos_encoding_3d = dict(
            type='Sine3D',
            embed_dims=_dim_,
            num_pt_feats=5,
            num_pos_feats=_dim_ // 2,
            temperature=10000,
            ),
        pos_emb_gen=None,
    ),
    pos_encoding_bev=None,

    head=dict(
        xs_loss_weight=2.0,
        zs_loss_weight=10.0,
        vis_loss_weight=1.0,
        cls_loss_weight=10,
        project_loss_weight=1.0,
        num_pt_per_line=num_pt_per_line,
        pred_dim=pred_dim,
        num_lidar_feat=num_lidar_feat - 1,
        insert_lidar_feat_before_img=True,
        neck = dict(
            type='FPN',
            in_channels=[256, 128, 256, 512],
            out_channels=_dim_,
            start_level=0,
            add_extra_convs='on_output',
            num_outs=4,
            relu_before_extra_convs=True
        ),
        ms2one=dict(
            type='DilateNaive',
            inc=_dim_, outc=_dim_, num_scales=4,
            dilations=(1, 2, 5, 9)
        ),
        depth_net=dict(
            in_channels=256,
            mid_channels=256,
            context_channels=256,
            depth_channels=None,
            position_range=position_range,
            use_bce_loss=True,
            depth_resolution=voxel_size[1],
            norm_gt_wo_mask=False
        ),
        sparse_ins_bev=Config(
            dict(
                encoder=dict(
                    out_dims=_dim_),
                decoder=dict(
                    num_query=num_query,
                    num_group=1,
                    with_pt_center_feats=False,
                    sparse_num_group=4,
                    hidden_dim=_dim_,
                    kernel_dim=_dim_,
                    num_classes=num_category,
                    num_convs=4,
                    output_iam=True,
                    scale_factor=1., 
                    ce_weight=2.0,
                    mask_weight=5.0,
                    dice_weight=2.0,
                    objectness_weight=1.0,
                ),
                sparse_decoder_weight=5.0,
        )),
    ),
    trans_params=dict(init_z=0, bev_h=250, bev_w=300), 
)

transformer=dict(
    type='DV3DLaneTransformer',
    decoder=dict(
        type='DV3DLaneTransformerDecoder',
        embed_dims=_dim_,
        num_layers=2,
        M_decay_ratio=1,
        num_query=num_query,
        num_anchor_per_query=num_pt_per_line,
        anchor_y_steps=anchor_y_steps,
        look_forward_twice=True,
        return_intermediate=False,
        transformerlayers=dict(
            type='DV3DLaneDecoderLayer',
            attn_cfgs=[
                dict(
                    type='MultiheadAttention',
                    embed_dims=_dim_,
                    num_heads=4,
                    dropout=0.1),
                dict(
                    type='MultiScaleDeformableAttention',
                    embed_dims=_dim_,
                    num_heads=4,
                    num_levels=1,
                    num_points=8,
                    batch_first=False,
                    dropout=0.1),
                ],
            ffn_cfgs=dict(
                type='FFN',
                embed_dims=_dim_,
                feedforward_channels=_dim_*8,
                num_fcs=2,
                ffn_drop=0.1,
                act_cfg=dict(type='ReLU', inplace=True),
            ),
            feedforward_channels=_dim_ * 8,
            operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                            'ffn', 'norm')),
))

sparse_ins_decoder=Config(
    dict(
        encoder=dict(
            out_dims=_dim_),
        decoder=dict(
            num_query=latr_cfg['num_query'],
            num_group=latr_cfg['num_group'],
            with_pt_center_feats=False,
            sparse_num_group=latr_cfg['sparse_num_group'],
            hidden_dim=_dim_,
            kernel_dim=_dim_,
            num_classes=num_category,
            num_convs=4,
            output_iam=True,
            scale_factor=1., 
            ce_weight=2.0,
            mask_weight=5.0,
            dice_weight=2.0,
            objectness_weight=1.0,
        ),
        sparse_decoder_weight=5.0,
))

resize_h = 360  # 与 image_size 匹配
resize_w = 480

nepochs = 48
eval_freq = 1
optimizer_cfg = dict(
    type='AdamW',
    lr=2e-4,
    betas=(0.95, 0.99),
    paramwise_cfg=dict(
        custom_keys={
            'sampling_offsets': dict(lr_mult=0.1),
        }),
    weight_decay=0.01)