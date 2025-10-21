import re
import os
import os.path as osp
import sys
import copy
import json
import glob
import pickle
import random
import warnings
import numpy as np
import cv2
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as F
from torchvision.transforms import InterpolationMode
from utils.utils import *
from experiments.gpu_utils import is_main_process
from mmdet3d.datasets.pipelines import (PointSample, PointShuffle, Compose, PointsRangeFilter)
from mmdet3d.core.points import LiDARPoints

from .transform import PhotoMetricDistortionMultiViewImage
from .lidar_utils import lidar2cam, cam2img, filter_fov, get_homo_coords
from .data_utils import smooth_lanes, near_one_pt, get_vis_mask

sys.path.append('./')
warnings.simplefilter('ignore', np.RankWarning)
matplotlib.use('Agg')

import yaml

class LaneDataset(Dataset):
    """
    Dataset with labeled lanes
        This implementation considers:
        w/o laneline 3D attributes
        w/o centerline annotations
        default considers 3D laneline, including centerlines

        This new version of data loader prepare ground-truth anchor tensor in flat ground space.
        It is assumed the dataset provides accurate visibility labels. Preparing ground-truth tensor depends on it.
    """
    # dataset_base_dir is image path, json_file_path is json file path,
    def __init__(self, dataset_base_dir, json_file_path, args, pipeline=None):
        """

        :param dataset_info_file: json file list
        """
        self.totensor = transforms.ToTensor()
        mean =args.mean
        std = args.std
        self.normalize = transforms.Normalize(mean, std)

        if pipeline is not None:
            img_pipes = []
            img_albu_pipe = None
            for pipe in pipeline['img_aug']:
                if pipe['type'] == 'Albu':
                    img_albu_pipe = Compose([pipe])
                else:
                    img_pipes.append(pipe)
            self.img_pipeline  = Compose(img_pipes)
            self.img_albu_pipe = img_albu_pipe
            self.pts_pipeline  = Compose(pipeline['pts_aug'])
            self.gt3d_pipeline = Compose(pipeline['gt3d_aug'])

        self.seg_bev = getattr(args, 'seg_bev', False)
        self.bev_thick = getattr(args, 'bev_thick', 2)
        self.front_thick = getattr(args, 'front_thick', 6)

        self.dataset_base_dir = dataset_base_dir #'./data/openlane/data_final/'
        self.json_file_path = json_file_path #'./data/openlane/lane3d_300/training or validation/'

        # dataset parameters
        self.dataset_name = args.dataset_name #'openlane'
        self.num_category = args.num_category #21

        self.h_org = args.org_h #1280
        self.w_org = args.org_w #1920
        self.h_crop = args.crop_y #0

        # parameters related to service network
        self.h_net = args.resize_h #720
        self.w_net = args.resize_w #960
        self.u_ratio = float(self.w_net) / float(self.w_org)
        self.v_ratio = float(self.h_net) / float(self.h_org - self.h_crop)
        self.top_view_region = args.top_view_region
        self.max_lanes = args.max_lanes #20

        self.K = args.K
        self.H_crop = homography_crop_resize([args.org_h, args.org_w], args.crop_y, [args.resize_h, args.resize_w])

        if args.fix_cam:
            self.fix_cam = True
            # compute the homography between image and IPM, and crop transformation
            self.cam_height = args.cam_height
            self.cam_pitch = np.pi / 180 * args.pitch
            self.P_g2im = projection_g2im(self.cam_pitch, self.cam_height, args.K)
        else:
            self.fix_cam = False

        # compute anchor steps
        self.use_default_anchor = args.use_default_anchor
        
        self.x_min, self.x_max = self.top_view_region[0, 0], self.top_view_region[1, 0] #(-10,10)
        self.y_min, self.y_max = self.top_view_region[2, 1], self.top_view_region[0, 1] #(3,103)
        
        self.anchor_y_steps = args.anchor_y_steps
        self.num_y_steps = len(self.anchor_y_steps) #20

        self.anchor_y_steps_dense = args.get(
            'anchor_y_steps_dense',
            np.linspace(3, 103, 200))
        args.anchor_y_steps_dense = self.anchor_y_steps_dense
        self.num_y_steps_dense = len(self.anchor_y_steps_dense) #200
        self.anchor_dim = 3 * self.num_y_steps + args.num_category
        self.save_json_path = args.save_json_path

        self.args = args
        self.waymo_veh2gd = np.array([
                                    [0, 1, 0],
                                    [-1, 0, 0],
                                    [0, 0, 1]], dtype=float)

        # parse ground-truth file
        if 'openlane' in self.dataset_name:
            label_list = sorted(glob.glob(json_file_path + '**/*.json', recursive=True)) #157807
            self._label_list = label_list
        elif 'once' in self.dataset_name:
            label_list = glob.glob(json_file_path + '*/*/*.json', recursive=True)
            self._label_list = []
            for js_label_file in label_list:
                if not os.path.getsize(js_label_file):
                    continue
                image_path = map_once_json2img(js_label_file)
                if not os.path.exists(image_path):
                    continue
                self._label_list.append(js_label_file)
        else: 
            raise ValueError("to use ApolloDataset for apollo")
        
        if hasattr(self, '_label_list'):
            self.n_samples = len(self._label_list)
        else:
            self.n_samples = self._label_image_path.shape[0]

    def preprocess_data_from_json_openlane(self, idx_json_file):
        _label_image_path = None
        _label_cam_height = None
        _label_cam_pitch = None
        cam_extrinsics = None
        cam_intrinsics = None
        _label_laneline_org = None
        _label_laneline_gd = None
        _gt_laneline_visibility = None
        _gt_laneline_category_org = None

        with open(idx_json_file, 'r') as file:
            info_dict = json.load(file)
            file_path = info_dict['file_path']  # e.g., training/segment-.../151865636859639400.jpg
            dir = osp.basename(osp.normpath(self.json_file_path)) #training/ or validation/'
            base_dir, filename = osp.split(file_path) # training/segment-.../ , 151865636859639400.jpg
            _ , seg_name = osp.split(base_dir) # segment-.../
            info_dict['file_path'] = osp.join(dir, seg_name, 'images', filename)
            image_path = osp.join(self.dataset_base_dir, info_dict['file_path'])
            assert ops.exists(image_path), '{:s} not exist'.format(image_path)
            _label_image_path = image_path

            # 推导点云路径：替换 images 为 velodyne_filter，.jpg 为 .bin
            point_cloud_filename = os.path.splitext(filename)[0] + '.bin'
            point_cloud_path = osp.join(self.dataset_base_dir, dir, seg_name, 'farthest_filter', point_cloud_filename)
            if osp.exists(point_cloud_path):
                _point_cloud_path = point_cloud_path
            else:
                print(f"警告：点云文件 {point_cloud_path} 不存在")

            if not self.fix_cam:
                T_cam2veh = np.array(info_dict['extrinsic'], dtype=np.float32)
                T_cam2gd = T_cam2veh.copy()
                T_cam2gd[:3, :3] = np.matmul(
                    np.linalg.inv(self.waymo_veh2gd), T_cam2gd[:3, :3])
                T_cam2gd[0:2, 3] = 0.0

                gt_cam_height = T_cam2gd[2, 3]
                if 'cam_pitch' in info_dict:
                    gt_cam_pitch = info_dict['cam_pitch']
                else:
                    gt_cam_pitch = 0

                cam_intrinsics = np.array(info_dict['intrinsic'])

            _label_cam_height = gt_cam_height
            _label_cam_pitch = gt_cam_pitch

            gt_lanes_packed_cam = info_dict['lane_lines']
            gt_lane_pts, gt_lane_visibility, gt_laneline_category = [], [], []
            for i, gt_lane_packed in enumerate(gt_lanes_packed_cam):
                # A GT lane can be either 2D or 3D
                # if a GT lane is 3D, the height is intact from 3D GT, so keep it intact here too
                lane = np.array(gt_lane_packed['xyz'])
                lane_visibility = np.array(gt_lane_packed['visibility'])

                # Coordinate convertion for openlane_300 data
                lane = np.vstack((lane, np.ones((1, lane.shape[1])))) 
                lane = np.matmul(T_cam2gd,  lane)

                lane = lane[0:3, :].T
                gt_lane_pts.append(lane)
                gt_lane_visibility.append(lane_visibility)

                if 'category' in gt_lane_packed:
                    lane_cate = gt_lane_packed['category']
                    if lane_cate == 21:  # merge left and right road edge into road edge
                        lane_cate = 20
                    gt_laneline_category.append(lane_cate)
                else:
                    gt_laneline_category.append(1)
        
        _gt_laneline_category_org = copy.deepcopy(np.array(gt_laneline_category))

        _label_laneline_gd = gt_lane_pts
        gt_visibility = gt_lane_visibility

        _label_laneline_gd = [prune_3d_lane_by_visibility(gt_lane_gd, gt_visibility[k]) for k, gt_lane_gd in enumerate(_label_laneline_gd)]
        _label_laneline_gd = [lane[lane[:,1].argsort()] for lane in _label_laneline_gd]
        T_cam2gd = T_cam2gd @ np.array(
            [[0, 0, 1, 0],
            [-1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, 0, 1]], dtype=float)

        processed_info_dict = {
           'label_image_path': _label_image_path,
           'label_cam_height': _label_cam_height,
           'label_cam_pitch' : _label_cam_pitch,
           'cam2gd'  : T_cam2gd,
           'cam2veh' : T_cam2veh,
           'cam_intrinsics' : cam_intrinsics,
           'label_laneline_gd' : _label_laneline_gd,
           'label_lanline_org' : gt_lanes_packed_cam,
           'gt_laneline_visibility' : _gt_laneline_visibility,
           'gt_laneline_category_org' : _gt_laneline_category_org,
           'point_cloud_path': point_cloud_path
        }
        return processed_info_dict, info_dict

    def __len__(self):
        """
        Conventional len method
        """
        return self.n_samples
    
    def process_points(self, 
                       points_all=None, 
                       T_cam2veh=None,
                       T_cam2gd=None,
                       intrinsics=None,
                       image=None
                       ):
        # lidar_pt = points_all.copy()
        R_vg = np.array([[0, 1, 0],
                        [-1, 0, 0],
                        [0, 0, 1]], dtype=float)
        R_gc = np.array([[1, 0, 0],
                        [0, 0, 1],
                        [0, -1, 0]], dtype=float)
        T_norm_cam2veh = T_cam2veh.copy()
        T_norm_cam2veh[:3, :3] = T_cam2veh[:3, :3] @ R_vg @ R_gc
        norm_cam_pt = lidar2cam(points_all, T_norm_cam2veh)
        img_pt, depth = cam2img(norm_cam_pt, intrinsics)
        img_pt, mask = filter_fov(img_pt, depth, image.height, image.width)

        # TODO lidar_encoding_manner: pt / voxel (pillar)
        lidar_fov_xyz = norm_cam_pt[mask].copy()
        lidar_fov_xyz = (
            T_cam2gd @ get_homo_coords(lidar_fov_xyz, with_return_T=True)).T #加1列，变齐次

        lidar_gd = lidar_fov_xyz[:, :3]
        max_points = 16384
        N, C = lidar_gd.shape
        if N < max_points:
            pad = np.zeros((max_points - N, C), dtype=lidar_gd.dtype)
            lidar_gd = np.vstack([lidar_gd, pad])

        return lidar_gd
    
    def gen_M(self, T_cam2gd, intrinsics, processed_info_dict, aug_mat=None):
        H_g2im, P_g2im, H_crop = self.transform_mats_impl(
            T_cam2gd,
            intrinsics, 
            processed_info_dict['label_cam_pitch'], 
            processed_info_dict['label_cam_height']
        )
        M = np.matmul(H_crop, P_g2im)
        # update transformation with image augmentation
        if aug_mat is not None:
            M = np.matmul(aug_mat, M)
        
        results = dict(
            H_g2im = H_g2im,
            P_g2im = P_g2im,
            H_crop = H_crop,
            # H_im2ipm = H_im2ipm,
            M = M
        )
        return results
    
    def gen_seg_labels(self, 
                       T_gd_label2img, 
                       gt_lanes_gd, 
                       gt_lanes_category, 
                       front_thick=6,
                       bev_thick=2,
                       points=None,
                       ):
        out = {}
        # prepare binary segmentation label map
        seg_label = np.zeros((self.h_net, self.w_net), dtype=np.int8)
        # seg idx has the same order as gt_lanes
        seg_idx_label = np.zeros((self.max_lanes, self.h_net, self.w_net), dtype=np.uint8)
        
        ground_lanes = np.zeros((self.max_lanes, self.anchor_dim), dtype=np.float32)
        ground_lanes_dense = np.zeros(
            (self.max_lanes, self.num_y_steps_dense * 3), dtype=np.float32)
        gt_laneline_img = [[0]] * len(gt_lanes_gd)

        if self.seg_bev:
            bev_seg_label = np.zeros(
                (self.args.grid_size[1], self.args.grid_size[0]),
                dtype=np.uint8)
            bev_seg_idx = np.zeros(
                (self.max_lanes, self.args.grid_size[1], self.args.grid_size[0]),
                dtype=np.uint8)
            assert points is not None
            single_points = points
            # max_v = single_points.max(axis=0)
            # min_v = single_points.min(axis=0)
            max_v = single_points.max(0)[0]
            min_v = single_points.min(0)[0]
            max_x, max_y = max_v[:2]
            min_x, min_y = min_v[:2]
            max_x_bev = (max_x - self.args.position_range[0]) / self.args.voxel_size[0]
            min_x_bev = (min_x - self.args.position_range[0]) / self.args.voxel_size[0]
            max_y_bev = (max_y - self.args.position_range[1]) / self.args.voxel_size[1]
            min_y_bev = (min_y - self.args.position_range[1]) / self.args.voxel_size[1]
            bev_seg_mask = np.zeros_like(bev_seg_label)
            bev_seg_mask[int(min_y_bev) : int(max_y_bev), int(min_x_bev) : int(max_x_bev)] = 1
            out['bev_seg_mask'] = bev_seg_mask

        seg_interp_label = seg_label.copy()
        for i, lane in enumerate(gt_lanes_gd):
            lane = lane.astype(np.float32)
            if i >= self.max_lanes:
                break

            # TODO remove this
            if lane.shape[0] <= 2:
                continue

            if gt_lanes_category[i] >= self.num_category:
                continue

            vis = get_vis_mask(self.anchor_y_steps, lane, tol_dist=5)
            xs, zs = resample_laneline_in_y(
                lane, self.anchor_y_steps,
                interp_kind='linear',
                outrange_use_polyfit=False)

            lane = np.stack([xs, self.anchor_y_steps, zs], axis=1)

            if vis.sum() < 2:
                continue
            
            x_2d, y_2d = projective_transformation(
                T_gd_label2img[:3],
                xs[vis], # xs_dense[vis_dense],
                self.anchor_y_steps[vis], # self.anchor_y_steps_dense[vis_dense],
                zs[vis], # zs_dense[vis_dense]
            )
            lane2d = np.stack([x_2d, y_2d], axis=-1)
            dec_idx = np.argsort(lane2d[:, 1])[::-1]
            lane2d = lane2d[dec_idx]
            if lane2d.shape[0] == 1 and near_one_pt(lane2d):
                seg_label = cv2.circle(seg_label, 
                                       (int(x_2d[0]), int(y_2d[0])), # tuple(map(np.int32, lane2d)),
                                       front_thick,
                                       1, # gt_lanes_category[i].item(),
                                       -1)
                seg_idx_label[i] = cv2.circle(seg_idx_label[i], 
                                              (int(x_2d[0]), int(y_2d[0])), # tuple(map(np.int32, lane2d)), 
                                              front_thick,
                                              gt_lanes_category[i].item(),
                                              -1)
            else:
                seg_label = cv2.polylines(
                    seg_label,
                    [np.int32(lane2d).reshape((-1, 1, 2))],
                    isClosed=False,
                    color=1,
                    thickness=front_thick
                )
                seg_idx_label[i] = cv2.polylines(
                    seg_idx_label[i],
                    [np.int32(lane2d).reshape((-1, 1, 2))],
                    isClosed=False,
                    color=gt_lanes_category[i].item(),
                    thickness=front_thick
                )
            
            if seg_idx_label[i].max() <= 0:
                continue
            
            if self.seg_bev:  #把真实车道线标签投影到 BEV 网格空间，并生成用于训练的 BEV 分割标签
                xs_bev = (xs - self.args.position_range[0]) / self.args.voxel_size[0]
                ys_bev = (self.anchor_y_steps - self.args.position_range[1]) / self.args.voxel_size[1]
                xs_bev_pos = xs_bev[vis]
                ys_bev_pos = ys_bev[vis]
                xs_bev_pos, ys_bev_pos = smooth_lanes(xs_bev_pos, ys_bev_pos)
                
                xs_bev_flipped = bev_seg_label.shape[1] - xs_bev_pos.astype(np.int32) - 1
                lane2d_bev = np.stack([xs_bev_flipped, ys_bev_pos.astype(np.int32)], axis=-1)
                bev_seg_label = cv2.polylines(bev_seg_label, [lane2d_bev], 
                                              isClosed=False, 
                                              color=gt_lanes_category[i].item(), thickness=bev_thick)
                bev_seg_idx[i] = cv2.polylines(bev_seg_idx[i], 
                                               [lane2d_bev], 
                                               isClosed=False, 
                                               color=gt_lanes_category[i].item(), thickness=bev_thick)
                
                if seg_idx_label[i].max() != bev_seg_idx[i].max():
                    seg_idx_label[i][:] = 0
                    bev_seg_idx[i][:] = 0
                    continue

            if bev_seg_label.max() <= 0:
                continue

            ground_lanes[i][0: self.num_y_steps] = xs
            ground_lanes[i][self.num_y_steps:2*self.num_y_steps] = zs
            ground_lanes[i][2*self.num_y_steps:3*self.num_y_steps] = vis * 1.0
            ground_lanes[i][self.anchor_dim - self.num_category] = 0.0
            ground_lanes[i][self.anchor_dim - self.num_category + gt_lanes_category[i]] = 1.0

            xs_dense, zs_dense = resample_laneline_in_y(
                lane, self.anchor_y_steps_dense,
                outrange_use_polyfit=True)
            vis_dense = np.logical_and(
                self.anchor_y_steps_dense > lane[:, 1].min(),
                self.anchor_y_steps_dense < lane[:, 1].max())
            ground_lanes_dense[i][0: self.num_y_steps_dense] = xs_dense
            ground_lanes_dense[i][1*self.num_y_steps_dense: 2*self.num_y_steps_dense] = zs_dense
            ground_lanes_dense[i][2*self.num_y_steps_dense: 3*self.num_y_steps_dense] = vis_dense * 1.0

        out.update(dict(
            seg_label = seg_label,
            seg_idx_label = seg_idx_label,
            bev_seg_label = bev_seg_label,
            ground_lanes = ground_lanes,
            ground_lanes_dense = ground_lanes_dense,
            bev_seg_idx=bev_seg_idx,
        ))
        return out

    # new getitem, WIP
    def WIP__getitem__(self, idx):
        """
        Args: idx (int): Index in list to load image
        """
        extra_dict = {}

        idx_json_file = self._label_list[idx]
        # preprocess data from json file
        if 'openlane' in self.dataset_name:
            processed_info_dict, json_info_dict = self.preprocess_data_from_json_openlane(idx_json_file)               
            
            T_cam2gd = processed_info_dict['cam2gd']
            T_cam2veh = processed_info_dict['cam2veh']
            img_name = processed_info_dict['label_image_path']
            point_cloud_path = processed_info_dict['point_cloud_path']

        # fetch camera height and pitch
        if not self.fix_cam:
            gt_cam_height = processed_info_dict['label_cam_height']
            gt_cam_pitch = processed_info_dict['label_cam_pitch']
            if 'openlane' in self.dataset_name or 'once' in self.dataset_name:
                intrinsics = processed_info_dict['cam_intrinsics']
                T_cam2gd = processed_info_dict['cam2gd']
                T_cam2veh = processed_info_dict['cam2veh']
            else:
                # should not be used
                intrinsics = self.K
                T_cam2gd = np.zeros((3,4))
                T_cam2gd[2,3] = gt_cam_height
        else:
            gt_cam_height = self.cam_height
            gt_cam_pitch = self.cam_pitch
            # should not be used
            intrinsics = self.K
            T_cam2gd = np.zeros((3,4))
            T_cam2gd[2,3] = gt_cam_height

        with open(img_name, 'rb') as f:
            image = (Image.open(f).convert('RGB'))

        # 加载点云数据并确保 float32
        point_cloud = None
        point_cloud = np.fromfile(point_cloud_path, dtype=np.float32).reshape(-1, 3)
        extra_dict['point_cloud'] = self.process_points(points_all=point_cloud,
                                         T_cam2veh=T_cam2veh,
                                         T_cam2gd=T_cam2gd,
                                         intrinsics=intrinsics,
                                         image=image)

        # image preprocess with crop and resize
        image = F.crop(image, self.h_crop, 0, self.h_org-self.h_crop, self.w_org)
        image = F.resize(image, size=(self.h_net, self.w_net), interpolation=InterpolationMode.BILINEAR)

        aug_mat = np.eye(3)
        if hasattr(self, 'img_pipeline'):
            if self.img_albu_pipe is not None:
                image = self.img_albu_pipe(dict(img=np.array(image)))['img']

            aug_dict = self.img_pipeline(dict(img=np.array(image)))
            image = Image.fromarray(
                np.clip(aug_dict['img'], 0, 255).astype(np.uint8))
            aug_mat = aug_dict.get('rot_mat', np.eye(3))

        trans = self.gen_M(T_cam2gd, intrinsics, processed_info_dict, aug_mat)
        T_gd_label2img = np.eye(4).astype(np.float32)
        T_gd_label2img[:3] = trans['M']

        intrinsics = np.matmul(self.H_crop, intrinsics)

        lidar_fov = extra_dict['point_cloud']
        lidar_fov = self.pts_pipeline(
                dict(points=LiDARPoints(lidar_fov, points_dim=lidar_fov.shape[1])))
        lidar_fov = lidar_fov['points'].tensor
        extra_dict['point_cloud'] = lidar_fov[:, :self.args.num_lidar_feat]

        seg_labels = self.gen_seg_labels(
            T_gd_label2img, 
            processed_info_dict['label_laneline_gd'], #车道真值
            processed_info_dict['gt_laneline_category_org'], #车道类别
            front_thick=self.front_thick,
            bev_thick=self.bev_thick,
            points=extra_dict['point_cloud']
        )

        image = self.totensor(image).float()
        image = self.normalize(image)
        intrinsics = torch.from_numpy(intrinsics).float()
        T_cam2gd = torch.from_numpy(T_cam2gd).float()
        T_cam2veh = torch.from_numpy(T_cam2veh).float()

        seg_label = torch.from_numpy(seg_labels['seg_label'].astype(np.float32))
        seg_label.unsqueeze_(0)

        extra_dict['seg'] = seg_label
        extra_dict['lane_idx'] = seg_labels['seg_idx_label']
        extra_dict['ground_lanes'] = seg_labels['ground_lanes']
        extra_dict['ground_lanes_dense'] = seg_labels['ground_lanes_dense']
        extra_dict['lidar2img'] = T_gd_label2img
        extra_dict['pad_shape'] = torch.Tensor(seg_labels['seg_idx_label'].shape[-2:]).float()
        extra_dict['idx_json_file'] = idx_json_file
        extra_dict['image'] = image
        extra_dict['intrinsics'] = intrinsics
        extra_dict['cam2lidar'] = T_cam2gd  # 不需调整
        # if self.data_aug:
        #     aug_mat = torch.from_numpy(aug_mat.astype(np.float32))
        #     extra_dict['aug_mat'] = aug_mat
        if self.seg_bev:
            extra_dict['bev_seg_idx_label'] = seg_labels['bev_seg_label']
            extra_dict['bev_seg_mask'] = seg_labels['bev_seg_mask']
            extra_dict['bev_seg_idx'] = seg_labels['bev_seg_idx']
        return extra_dict

    # old getitem, workable
    def __getitem__(self, idx):
        """
        Args: idx (int): Index in list to load image
        """
        return self.WIP__getitem__(idx)

    def transform_mats_impl(self, cam_extrinsics, cam_intrinsics, cam_pitch, cam_height):
        H_g2im = homograpthy_g2im_extrinsic(cam_extrinsics, cam_intrinsics)
        P_g2im = projection_g2im_extrinsic(cam_extrinsics, cam_intrinsics)

        # H_im2ipm = np.linalg.inv(np.matmul(self.H_crop, np.matmul(H_g2im, self.H_ipm2g))).astype(np.float32)
        
        return H_g2im, P_g2im, self.H_crop

def make_lane_y_mono_inc(lane):
    """
        Due to lose of height dim, projected lanes to flat ground plane may not have monotonically increasing y.
        This function trace the y with monotonically increasing y, and output a pruned lane
    :param lane:
    :return:
    """
    idx2del = []
    max_y = lane[0, 1]
    for i in range(1, lane.shape[0]):
        # hard-coded a smallest step, so the far-away near horizontal tail can be pruned
        if lane[i, 1] <= max_y + 3:
            idx2del.append(i)
        else:
            max_y = lane[i, 1]
    lane = np.delete(lane, idx2del, 0)
    return lane

def data_aug_rotate(img):
    # assume img in PIL image format
    rot = random.uniform(-np.pi/18, np.pi/18)
    center_x = img.width / 2
    center_y = img.height / 2
    rot_mat = cv2.getRotationMatrix2D((center_x, center_y), rot, 1.0)
    img_rot = np.array(img)
    img_rot = cv2.warpAffine(img_rot, rot_mat, (img.width, img.height), flags=cv2.INTER_LINEAR)
    rot_mat = np.vstack([rot_mat, [0, 0, 1]])
    return img_rot, rot_mat


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_loader(transformed_dataset, args):
    """
        create dataset from ground-truth
        return a batch sampler based ont the dataset
    """

    # transformed_dataset = LaneDataset(dataset_base_dir, json_file_path, args)
    sample_idx = range(transformed_dataset.n_samples)

    g = torch.Generator()
    g.manual_seed(0)

    discarded_sample_start = len(sample_idx) // args.batch_size * args.batch_size
    if is_main_process():
        print("Discarding images:")
        if hasattr(transformed_dataset, '_label_image_path'):
            print(transformed_dataset._label_image_path[discarded_sample_start: len(sample_idx)])
        else:
            print(len(sample_idx) - discarded_sample_start)
    sample_idx = sample_idx[0 : discarded_sample_start]
    
    if args.dist:
        if is_main_process():
            print('use distributed sampler')
        if 'standard' in args.dataset_name or 'rare_subset' in args.dataset_name or 'illus_chg' in args.dataset_name:
            data_sampler = torch.utils.data.distributed.DistributedSampler(transformed_dataset, shuffle=True, drop_last=True)
            data_loader = DataLoader(transformed_dataset,
                                        batch_size=args.batch_size, 
                                        sampler=data_sampler,
                                        num_workers=args.nworkers, 
                                        pin_memory=True,
                                        persistent_workers=args.nworkers > 0,
                                        worker_init_fn=seed_worker,
                                        generator=g,
                                        drop_last=True)
        else:
            data_sampler = torch.utils.data.distributed.DistributedSampler(transformed_dataset)
            data_loader = DataLoader(transformed_dataset,
                                        batch_size=args.batch_size, 
                                        sampler=data_sampler,
                                        num_workers=args.nworkers, 
                                        pin_memory=True,
                                        persistent_workers=args.nworkers > 0,
                                        worker_init_fn=seed_worker,
                                        generator=g)
    else:
        if is_main_process():
            print("use default sampler")
        data_sampler = torch.utils.data.sampler.SubsetRandomSampler(sample_idx)
        data_loader = DataLoader(transformed_dataset,
                                batch_size=args.batch_size, sampler=data_sampler,
                                num_workers=args.nworkers, pin_memory=True,
                                persistent_workers=args.nworkers > 0,
                                worker_init_fn=seed_worker,
                                generator=g)

    if args.dist:
        return data_loader, data_sampler
    return data_loader

def map_once_json2img(json_label_file):
    if 'train' in json_label_file:
        split_name = 'train'
    elif 'val' in json_label_file:
        split_name = 'val'
    elif 'test' in json_label_file:
        split_name = 'test'
    else:
        raise ValueError("train/val/test not in the json path")
    image_path = json_label_file.replace(split_name, 'data').replace('.json', '.jpg')
    return image_path
