# modify from https://github.com/mit-han-lab/bevfusion
from typing import Tuple
import torch
from torch import nn
import numpy as np
from scipy.spatial import cKDTree
from .ops import bev_pool
from mmdet.models.builder import NECKS


def gen_dx_bx(xbound, ybound, zbound):
    dx = torch.Tensor([row[2] for row in [xbound, ybound, zbound]])
    bx = torch.Tensor(
        [row[0] + row[2] / 2.0 for row in [xbound, ybound, zbound]])
    nx = torch.LongTensor([(row[1] - row[0]) / row[2]
                           for row in [xbound, ybound, zbound]])
    return dx, bx, nx


class BaseViewTransform(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        image_size: Tuple[int, int],
        feature_size: Tuple[int, int],
        xbound: Tuple[float, float, float],
        ybound: Tuple[float, float, float],
        zbound: Tuple[float, float, float],
        dbound: Tuple[float, float, float],
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.image_size = image_size
        self.feature_size = feature_size
        self.xbound = xbound
        self.ybound = ybound
        self.zbound = zbound
        self.dbound = dbound

        dx, bx, nx = gen_dx_bx(self.xbound, self.ybound, self.zbound)
        self.dx = nn.Parameter(dx, requires_grad=False)
        self.bx = nn.Parameter(bx, requires_grad=False)
        self.nx = nn.Parameter(nx, requires_grad=False)

        self.C = out_channels
        self.frustum = self.create_frustum()
        self.D = self.frustum.shape[0]
        self.fp16_enabled = False

    def create_frustum(self):
        iH, iW = self.image_size #360, 480
        fH, fW = self.feature_size #45, 60

        ds = (
            torch.arange(*self.dbound,
                         dtype=torch.float).view(-1, 1, 1).expand(-1, fH, fW))
        D, _, _ = ds.shape

        xs = (
            torch.linspace(0, iW - 1, fW,
                           dtype=torch.float).view(1, 1, fW).expand(D, fH, fW))
        ys = (
            torch.linspace(0, iH - 1, fH,
                           dtype=torch.float).view(1, fH, 1).expand(D, fH, fW))

        frustum = torch.stack((xs, ys, ds), -1)
        return nn.Parameter(frustum, requires_grad=False)

    def get_geometry(
        self,
        camera2lidar_rots,
        camera2lidar_trans,
        intrins,
        **kwargs,
    ):
        B, _ = camera2lidar_trans.shape

        # undo post-transformation
        # B x N x D x H x W x 3  (158,45,60个格子的长方体)相机坐标系，3的值表示真实像素位置值
        points = self.frustum
        points = points.unsqueeze(0).unsqueeze(-1)

        # cam_to_lidar
        points = torch.cat(
            (
                points[:, :, :, :, :2] * points[:, :, :, :,  2:3],
                points[:, :, :, :, 2:3],
            ),
            4,
        )
        combine = camera2lidar_rots.matmul(torch.inverse(intrins))
        points = combine.view(B, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        points += camera2lidar_trans.view(B, 1, 1, 1, 3)

        # if 'extra_rots' in kwargs:
        #     extra_rots = kwargs['extra_rots']
        #     points = (
        #         extra_rots.view(B, 1, 1, 1, 1, 3,
        #                         3).repeat(1, N, 1, 1, 1, 1, 1).matmul(
        #                             points.unsqueeze(-1)).squeeze(-1))
        # if 'extra_trans' in kwargs:
        #     extra_trans = kwargs['extra_trans']
        #     points += extra_trans.view(B, 1, 1, 1, 1,
        #                                3).repeat(1, N, 1, 1, 1, 1)

        return points

    def get_cam_feats(self, x):
        raise NotImplementedError

    def bev_pool(self, geom_feats, x):
        B,  D, H, W, C = x.shape
        Nprime = B * D * H * W

        # flatten x
        x = x.reshape(Nprime, C)

        # flatten indices
        geom_feats = ((geom_feats - (self.bx - self.dx / 2.0)) /
                      self.dx).long()
        geom_feats = geom_feats.view(Nprime, 3)
        batch_ix = torch.cat([
            torch.full([Nprime // B, 1], ix, device=x.device, dtype=torch.long)
            for ix in range(B)
        ])
        geom_feats = torch.cat((geom_feats, batch_ix), 1) #(x_idx, y_idx, z_idx, batch_idx)

        # filter out points that are outside box
        kept = ((geom_feats[:, 0] >= 0)
                & (geom_feats[:, 0] < self.nx[0])
                & (geom_feats[:, 1] >= 0)
                & (geom_feats[:, 1] < self.nx[1])
                & (geom_feats[:, 2] >= 0)
                & (geom_feats[:, 2] < self.nx[2]))
        x = x[kept]
        geom_feats = geom_feats[kept]

        x = bev_pool(x, geom_feats, B, self.nx[2], self.nx[0], self.nx[1])

        # collapse Z
        final = torch.cat(x.unbind(dim=2), 1)

        return final

    def forward(
        self,
        img,
        points,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        metas,
        **kwargs,
    ):
        intrins = camera_intrinsics[..., :3, :3]
        post_rots = img_aug_matrix[..., :3, :3]
        post_trans = img_aug_matrix[..., :3, 3]
        camera2lidar_rots = camera2lidar[..., :3, :3]
        camera2lidar_trans = camera2lidar[..., :3, 3]

        extra_rots = lidar_aug_matrix[..., :3, :3]
        extra_trans = lidar_aug_matrix[..., :3, 3]

        geom = self.get_geometry(
            camera2lidar_rots,
            camera2lidar_trans,
            intrins,
            post_rots,
            post_trans,
            extra_rots=extra_rots,
            extra_trans=extra_trans,
        )

        x = self.get_cam_feats(img)
        x = self.bev_pool(geom, x)
        return x


class LSSTransform(BaseViewTransform):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        image_size: Tuple[int, int],
        feature_size: Tuple[int, int],
        xbound: Tuple[float, float, float],
        ybound: Tuple[float, float, float],
        zbound: Tuple[float, float, float],
        dbound: Tuple[float, float, float],
        downsample: int = 1,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            image_size=image_size,
            feature_size=feature_size,
            xbound=xbound,
            ybound=ybound,
            zbound=zbound,
            dbound=dbound,
        )
        self.depthnet = nn.Conv2d(in_channels, self.D + self.C, 1)
        if downsample > 1:
            assert downsample == 2, downsample
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
                nn.Conv2d(
                    out_channels,
                    out_channels,
                    3,
                    stride=downsample,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
                nn.Conv2d(
                    out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
            )
        else:
            self.downsample = nn.Identity()

    def get_cam_feats(self, x):
        B, N, C, fH, fW = x.shape

        x = x.view(B * N, C, fH, fW)

        x = self.depthnet(x)
        depth = x[:, :self.D].softmax(dim=1)
        x = depth.unsqueeze(1) * x[:, self.D:(self.D + self.C)].unsqueeze(2)

        x = x.view(B, N, self.C, self.D, fH, fW)
        x = x.permute(0, 1, 3, 4, 5, 2)
        return x

    def forward(self, *args, **kwargs):
        x = super().forward(*args, **kwargs)
        x = self.downsample(x)
        return x


class BaseDepthTransform(BaseViewTransform):

    def forward(
        self,
        img,
        points,
        lidar2image,
        cam_intrinsic,
        camera2lidar,
        **kwargs,
    ):
        intrins = cam_intrinsic[:, :3, :3]
        camera2lidar_rots = camera2lidar[:, :3, :3]
        camera2lidar_trans = camera2lidar[:, :3, 3]

        batch_size = len(img)
        depth = torch.zeros(batch_size, 1, *self.image_size).to(points[0].device)

        for b in range(batch_size):
            cur_coords = points[b][:, :3]
            cur_lidar2image = lidar2image[b]
            cur_coords = cur_coords.transpose(1, 0)

            # lidar2image
            cur_coords = cur_lidar2image[:3, :3].matmul(cur_coords)
            cur_coords += cur_lidar2image[:3, 3].reshape(3, 1)
            # get 2d coords
            dist = cur_coords[2, :]
            cur_coords[2, :].clamp_(1e-5, 1e5)
            cur_coords[:2, :] /= cur_coords[2:3, :]

            # 检查点是否在图像范围内
            cur_coords = cur_coords[:2, :].transpose(0, 1)
            cur_coords = cur_coords[..., [1, 0]]  # 将 (x, y) 转换为图像坐标系  (x, y) -> (y, x)
            on_img = ((cur_coords[:, 0] < self.image_size[0])
                      & (cur_coords[:, 0] >= 0)
                      & (cur_coords[:, 1] < self.image_size[1])
                      & (cur_coords[:, 1] >= 0))
            # print(sum(on_img[0]))
            masked_coords = cur_coords[on_img].long()
            masked_dist = dist[on_img]
            depth = depth.to(masked_dist.dtype)
            depth[b, 0, masked_coords[:, 0],
                    masked_coords[:, 1]] = masked_dist

        geom = self.get_geometry(
            camera2lidar_rots,
            camera2lidar_trans,
            intrins,
        )
        
        x = self.get_cam_feats(img, depth)
        x = self.bev_pool(geom, x)
        return x

@NECKS.register_module()
class DepthLSSTransform(BaseDepthTransform):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        image_size: Tuple[int, int],
        feature_size: Tuple[int, int],
        xbound: Tuple[float, float, float],
        ybound: Tuple[float, float, float],
        zbound: Tuple[float, float, float],
        dbound: Tuple[float, float, float],
        downsample: int = 1,
    ) -> None:
        """Compared with `LSSTransform`, `DepthLSSTransform` adds sparse depth
        information from lidar points into the inputs of the `depthnet`."""
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            image_size=image_size,
            feature_size=feature_size,
            xbound=xbound,
            ybound=ybound,
            zbound=zbound,
            dbound=dbound,
        )
        self.dtransform = nn.Sequential(
            nn.Conv2d(1, 8, 1),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.Conv2d(8, 32, 5, stride=4, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )
        self.depthnet = nn.Sequential(
            nn.Conv2d(in_channels + 64, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
            nn.Conv2d(in_channels, self.D + self.C, 1),
        )
        if downsample > 1:
            assert downsample == 2, downsample
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
                nn.Conv2d(
                    out_channels,
                    out_channels,
                    3,
                    stride=downsample,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
                nn.Conv2d(
                    out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
            )
        else:
            self.downsample = nn.Identity()

    def get_cam_feats(self, x, d):
        B, C, fH, fW = x.shape

        d = d.view(B, 1, *d.shape[2:])
        x = x.view(B, C, fH, fW)

        d = self.dtransform(d)
        x = torch.cat([d, x], dim=1)
        x = self.depthnet(x)

        depth = x[:, :self.D].softmax(dim=1)
        x = depth.unsqueeze(1) * x[:, self.D:(self.D + self.C)].unsqueeze(2)  #深度概率加权

        x = x.view(B, self.C, self.D, fH, fW)
        x = x.permute(0, 2, 3, 4, 1) #torch.Size([1, 200, 45, 60, 80])
        return x

    def forward(self, *args, **kwargs):
        x = super().forward(*args, **kwargs)
        self.downsample = self.downsample.to('cuda')
        x = self.downsample(x)
        return x
    

@NECKS.register_module()
class GraphDepthLSSTransform(BaseViewTransform):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        image_size: tuple,
        feature_size: tuple,
        xbound: tuple,
        ybound: tuple,
        zbound: tuple,
        dbound: tuple,
        downsample: int = 1,
        K_graph: int = 8,
        noise: bool = False
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            image_size=image_size,
            feature_size=feature_size,
            xbound=xbound,
            ybound=ybound,
            zbound=zbound,
            dbound=dbound,
        )
        self.K_graph = K_graph
        self.noise = noise
        self.downsample_ratio = downsample

        # --- GraphBEV 特有网络层 ---
        self.dtransform = nn.Sequential(
            nn.Conv2d(16, 32, 5, stride=4, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )
        self.dtransform_Conv1 = nn.Sequential(
            nn.Conv2d(1, 8, 1),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
        )
        self.dtransform_Conv2 = nn.Sequential(
            nn.Conv2d(self.K_graph, 8, 1),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
        )
        self.depthnet = nn.Sequential(
            nn.Conv2d(in_channels + 64, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
            nn.Conv2d(in_channels, self.D + self.C, 1),
        )

        if downsample > 1:
            assert downsample == 2, downsample
            self.downsample = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
                nn.Conv2d(out_channels, out_channels, 3, stride=downsample, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
            )
        else:
            self.downsample = nn.Identity()

    def dual_input_dtransform(self, input1, input2):
        x1 = self.dtransform_Conv1(input1)
        x2 = self.dtransform_Conv2(input2)
        x = torch.cat([x1, x2], dim=1)
        x = self.dtransform(x)
        return x

    def cKDTree_neighbor(self, masked_coords):
        """
        修正版：返回邻居索引而非坐标，用于 Gather 操作
        """
        masked_coords_numpy = masked_coords.cpu().numpy()
        
        # 1. 去重 (必须做，否则 KDTree 计算距离为0)
        unique_coords, unique_indices = np.unique(masked_coords_numpy, axis=0, return_index=True)

        # 2. 建树与查询 (查 K+1 个，包含自己)
        kdtree = cKDTree(unique_coords)
        _, neighbor_indices = kdtree.query(unique_coords, k=self.K_graph + 1)
        
        # 3. 边界检查 (防止索引越界)
        max_len = len(unique_coords)
        neighbor_indices[neighbor_indices >= max_len] = 0

        # 4. 【关键修正】去掉第一列(自己)，保留后 K 个邻居
        neighbor_indices = neighbor_indices[:, 1:] 
        
        # 5. 转回 Tensor
        # neighbor_indices_torch: [M_unique, K] -> 邻居在 unique 数组里的下标
        neighbor_indices_torch = torch.from_numpy(neighbor_indices).long().to(masked_coords.device)
        
        # unique_coords_torch: [M_unique, 2] -> 去重后的坐标 (用于填回 tensor)
        unique_coords_torch = torch.from_numpy(unique_coords).long().to(masked_coords.device)
        
        # unique_original_indices: [M_unique] -> 去重后的点在原始 masked_dist 里的下标 (用于获取深度)
        unique_original_indices = torch.from_numpy(unique_indices).long().to(masked_coords.device)

        return neighbor_indices_torch, unique_coords_torch, unique_original_indices

    def get_cam_feats(self, img, depth, neighbors_depth):
        B, C, fH, fW = img.shape #[1, 256, 45, 60]
        depth = depth.view(B , 1,  *depth.shape[2:]) #[1, 1, 360, 480]
        neighbors_depth = neighbors_depth.view(B , *neighbors_depth.shape[1:]) #[1, 8, 360, 480]
        img = img.view(B , C, fH, fW)

        # Graph 特征提取
        depth_feat = self.dual_input_dtransform(depth, neighbors_depth)
        img = torch.cat([depth_feat, img], dim=1)
        img = self.depthnet(img)

        # 临时保存 Depth Logits 用于可视化 (保存 Batch 0, View 0)
        # img shape: [B*N, D+C_context, fH, fW]
        # self.D 是深度分类的数量 (bins)
        # if not hasattr(self, 'has_saved_vis'): # 这是一个简单的flag，防止每个iter都保存
        #      # 提取深度部分的 Logits
        #     logits_save = img[0, :self.D].detach().cpu().numpy() 
        #     np.save('temp_depth_logits.npy', logits_save)
        #     print(">>> [DEBUG] Saved temp_depth_logits.npy for visualization")
        #     # self.has_saved_vis = True # 如果只想保存第一帧就取消注释，否则会一直覆盖

        depth_prob = img[:, :self.D].softmax(dim=1)
        context = img[:, self.D : (self.D + self.C)]
        
        img = depth_prob.unsqueeze(1) * context.unsqueeze(2) 

        # 输出形状 [B, D, H, W, C]
        img = img.view(B, self.C, self.D, fH, fW)
        img = img.permute(0, 2, 3, 4, 1)
        return img

    def forward(
        self, 
        img, 
        points, 
        lidar2image, 
        cam_intrinsic, 
        camera2lidar, 
        metas=None, 
        **kwargs
    ):
        intrins = cam_intrinsic[:, :3, :3]
        camera2lidar_rots = camera2lidar[:, :3, :3]
        camera2lidar_trans = camera2lidar[:, :3, 3]

        batch_size = len(img)
        depth = torch.zeros(batch_size, 1, *self.image_size).to(points[0].device)
        neighbors_depth = torch.zeros(batch_size, self.K_graph, *self.image_size).to(points[0].device)

        for b in range(batch_size):
            cur_coords = points[b][:, :3]
            cur_lidar2image = lidar2image[b]
            cur_coords = cur_coords.transpose(1, 0)

            # lidar2image
            cur_coords = cur_lidar2image[:3, :3].matmul(cur_coords)
            cur_coords += cur_lidar2image[:3, 3].reshape(3, 1)
            # get 2d coords
            dist = cur_coords[2, :]
            cur_coords[2, :].clamp_(1e-5, 1e5)
            cur_coords[:2, :] /= cur_coords[2:3, :]

            # 检查点是否在图像范围内
            cur_coords = cur_coords[:2, :].transpose(0, 1)
            cur_coords = cur_coords[..., [1, 0]]
            on_img = ((cur_coords[:, 0] < self.image_size[0])
                      & (cur_coords[:, 0] >= 0)
                      & (cur_coords[:, 1] < self.image_size[1])
                      & (cur_coords[:, 1] >= 0))
            
            # 1. 获取有效点和深度
            masked_coords = cur_coords[on_img].long()
            masked_dist = dist[on_img]

            if masked_coords.shape[0] <= self.K_graph:
                continue

            # 2. 调用修正后的 KDTree 函数
            # n_idx: [M, K] 邻居下标
            # u_coords: [M, 2] 去重后的中心点坐标
            # u_orig_idx: [M] 去重后的点在原始 masked_dist 里的下标
            n_idx, u_coords, u_orig_idx = self.cKDTree_neighbor(masked_coords)

            # 3. 准备数据：只取去重后的点的深度
            unique_dist = masked_dist[u_orig_idx] 

            # 4. 填充中心深度 (Center Depth)
            # 把 unique_dist 填入中心点 u_coords 的位置
            depth[b, 0, u_coords[:, 0], u_coords[:, 1]] = unique_dist

            # 5. 【关键修改】Gather 邻居深度
            # 用 n_idx 去查 unique_dist，得到每个中心点对应的 K 个邻居的深度
            # neighbor_vals shape: [M, K]
            neighbor_vals = unique_dist[n_idx] 

            # 6. 【关键修改】填充邻居深度 (Assignment)
            # 逻辑：在 u_coords (中心点) 的位置，填入 neighbor_vals (邻居的值)
            # 利用 Broadcasting 避免循环，将 K 个邻居深度分别填入 K 个通道
            for k in range(self.K_graph):
                neighbors_depth[b, k, u_coords[:, 0], u_coords[:, 1]] = neighbor_vals[:, k]

        geom = self.get_geometry(
            camera2lidar_rots, 
            camera2lidar_trans, 
            intrins, 
        )
        
        x = self.get_cam_feats(img, depth, neighbors_depth)
        x = self.bev_pool(geom, x)
        x = self.downsample(x)
        return x
