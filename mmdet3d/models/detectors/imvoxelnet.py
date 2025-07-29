# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple, Union

import torch
import torch.nn.functional as F
from mmengine.structures import InstanceData
from PIL import Image
import torchvision.transforms as T

from mmdet3d.models.detectors import Base3DDetector
from mmdet3d.models.layers.fusion_layers.point_fusion import point_sample
from mmdet3d.registry import MODELS, TASK_UTILS
from mmdet3d.structures.bbox_3d import get_proj_mat_by_coord_type
from mmdet3d.structures.det3d_data_sample import SampleList
from mmdet3d.utils import ConfigType, OptConfigType, OptInstanceList
import numpy as np
import open3d as o3d
import os

@MODELS.register_module()
class ImVoxelNet(Base3DDetector):
    r"""`ImVoxelNet <https://arxiv.org/abs/2106.01178>`_.

    Args:
        backbone (:obj:`ConfigDict` or dict): The backbone config.
        neck (:obj:`ConfigDict` or dict): The neck config.
        neck_3d (:obj:`ConfigDict` or dict): The 3D neck config.
        bbox_head (:obj:`ConfigDict` or dict): The bbox head config.
        prior_generator (:obj:`ConfigDict` or dict): The prior points
            generator config.
        n_voxels (list): Number of voxels along x, y, z axis.
        coord_type (str): The type of coordinates of points cloud:
            'DEPTH', 'LIDAR', or 'CAMERA'.
        train_cfg (:obj:`ConfigDict` or dict, optional): Config dict of
            training hyper-parameters. Defaults to None.
        test_cfg (:obj:`ConfigDict` or dict, optional): Config dict of test
            hyper-parameters. Defaults to None.
        data_preprocessor (dict or ConfigDict, optional): The pre-process
            config of :class:`BaseDataPreprocessor`.  it usually includes,
                ``pad_size_divisor``, ``pad_value``, ``mean`` and ``std``.
        init_cfg (:obj:`ConfigDict` or dict, optional): The initialization
            config. Defaults to None.
    """

    def __init__(self,
                 backbone: ConfigType,
                 neck: ConfigType,
                 neck_3d: ConfigType,
                 bbox_head: ConfigType,
                 prior_generator: ConfigType,
                 n_voxels: List,
                 coord_type: str,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptConfigType = None):
        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        self.backbone = MODELS.build(backbone)
        self.neck = MODELS.build(neck)
        self.neck_3d = MODELS.build(neck_3d)
        bbox_head.update(train_cfg=train_cfg)
        bbox_head.update(test_cfg=test_cfg)
        self.bbox_head = MODELS.build(bbox_head)
        self.prior_generator = TASK_UTILS.build(prior_generator)
        self.n_voxels = n_voxels
        self.coord_type = coord_type
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def save_pointcloud_from_voxels(self, volume, valid_mask, voxel_size=(1,1,1), point_cloud_range=(-10, -10, -10, 10, 10, 10), filename='scene.ply'):
        """
        Save non-zero voxels as a 3D point cloud in PLY format.

        Args:
            volume (torch.Tensor): (C, Z, Y, X)
            valid_mask (torch.Tensor): (1, Z, Y, X)
            voxel_size (tuple): (vx, vy, vz)
            point_cloud_range (tuple): (x_min, y_min, z_min, x_max, y_max, z_max)
            filename (str): Output PLY file name
        """
        volume = volume.cpu().detach().numpy()
        valid_mask = valid_mask[0].cpu().detach().numpy()

        zyx_indices = np.stack(np.nonzero(valid_mask), axis=-1)
        if len(zyx_indices) == 0:
            print("No valid points to save.")
            return

        voxel_origin = np.array(point_cloud_range[:3])
        voxel_size = np.array(voxel_size)

        xyz_points = zyx_indices[:, [2, 1, 0]] * voxel_size + voxel_origin

        mean_volume = volume.mean(0)
        valid_vals = mean_volume[valid_mask]

        vmin, vmax = valid_vals.min(), valid_vals.max()

        if vmin == vmax:
            print("Warning: Flat volume, all intensities are the same.")
            normalized = np.zeros_like(valid_vals)
        else:
            normalized = (valid_vals - vmin) / (vmax - vmin)

        colors = np.stack([
            1 - normalized,
            normalized,
            np.zeros_like(normalized)
        ], axis=-1)

        keep_mask = (normalized < 1.0) & (normalized > 0.3)

        xyz_points = xyz_points[keep_mask]
        colors = colors[keep_mask]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz_points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        o3d.io.write_point_cloud(filename, pcd)

    def extract_feat(self, batch_inputs_dict: dict, batch_data_samples: SampleList):
        """Extract 3d features from the backbone -> fpn -> 3d projection.

        -> 3d neck -> bbox_head.

        Args:
            batch_inputs_dict (dict): The model input dict which include
                the 'imgs' key.

                    - imgs (torch.Tensor, optional): Image of each sample.
            batch_data_samples (list[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            Tuple:
            - torch.Tensor: Features of shape (N, C_out, N_x, N_y, N_z).
            - torch.Tensor: Valid mask of shape (N, 1, N_x, N_y, N_z).
        """
        if "imgs" in batch_inputs_dict.keys():
            imgs = batch_inputs_dict['imgs'].unsqueeze(1)
        else:
            imgs = batch_inputs_dict['img']
            imgs = torch.stack(imgs, dim=0)

        batch_img_metas = [
            data_samples.metainfo for data_samples in batch_data_samples
        ]

        batch_size, n_views = len(imgs), len(imgs[0])
        all_volumes = [[] for _ in range(batch_size)]
        all_valid_preds = [[] for _ in range(batch_size)]

        # iterate over each view (within the batch), each iteration yields 4 sets of points
        for i in range(n_views):
            img = imgs[:, i]
            img = img[:, :3].float()
            x = self.backbone(img)
            x = self.neck(x)[0]
            points = self.prior_generator.grid_anchors([self.n_voxels[::-1]], device=img.device)[0][:, :3]
        
            for b in range(batch_size):
                sharpen_filter = SharpeningFilter(x[b].shape[0]).to(x.device)
                sharpend_feat = sharpen_filter(x[b][None, ...])

                img_meta = batch_img_metas[b]

                scale = img_meta.get('scale_factor', 1)
                if isinstance(scale, (list, tuple)):
                    img_scale_factor = torch.tensor(scale[:2], dtype=points.dtype, device=points.device)
                else:
                    img_scale_factor = torch.tensor([scale, scale], dtype=points.dtype, device=points.device)

                img_flip = img_meta.get('flip', False)
                img_crop_offset = torch.tensor(img_meta.get('img_crop_offset', 0), dtype=points.dtype, device=points.device)
                if "imgs" in batch_inputs_dict.keys():
                    proj_mat = get_proj_mat_by_coord_type(img_meta, self.coord_type)
                else:
                    proj_mat = get_proj_mat_by_coord_type(img_meta, self.coord_type)[i]


                proj_mat = torch.tensor(proj_mat, dtype=points.dtype, device=points.device)
                volume = point_sample(
                    img_meta,
                    img_features=sharpend_feat,
                    points=points,
                    proj_mat=points.new_tensor(proj_mat),
                    coord_type=self.coord_type,
                    img_scale_factor=img_scale_factor,
                    img_crop_offset=img_crop_offset,
                    img_flip=img_flip,
                    img_pad_shape=img.shape[-2:],
                    img_shape=img_meta['img_shape'][:2],
                    aligned=False
                )
                volume = volume.reshape(self.n_voxels[::-1] + [-1]).permute(3, 2, 1, 0)
                all_valid_preds[b].append(~torch.all(volume == 0, dim=0, keepdim=True))
                all_volumes[b].append(volume)

        fused_volumes = []
        valid_preds = []

        for vols, preds in zip(all_volumes, all_valid_preds):
            vols = torch.stack(vols, dim=0)        # [T, C, Z, Y, X]
            preds = torch.stack(preds, dim=0)      # [T, 1, Z, Y, X]
            preds = preds.float()

            masked_vols = vols * preds             # [T, C, Z, Y, X]
            valid_count = preds.sum(dim=0)         # [1, Z, Y, X]
            valid_count[valid_count == 0] = 1 

            fused_volume = masked_vols.sum(dim=0) / n_views

            final_valid_mask = valid_count > 0
            fused_volume[:, ~final_valid_mask[0]] = 0

            fused_volumes.append(fused_volume)
            valid_preds.append(final_valid_mask)


        if "imgs" in batch_inputs_dict.keys():
            img_filepath = batch_img_metas[0]['img_path']
        else:
            img_filepath = batch_img_metas[0]['img_path'][0]

        if "images" in img_filepath:
            dataset_path = f"{img_filepath.split('images')[0]}/pc_vis/"
        else:
            dataset_path = f"{img_filepath.split('image')[0]}/pc_vis/"

        pc_filepath = f"{dataset_path}/{img_filepath.split('/')[-1].split('.')[0]}.ply"
        os.makedirs(dataset_path, exist_ok=True) 
        self.save_pointcloud_from_voxels(
            fused_volumes[0],
            valid_preds[0],
            filename=pc_filepath
        )

        if "images" in img_filepath:
            feature_map_dir = f"{img_filepath.split('images')[0]}/fm_vis/"
        else:
            feature_map_dir = f"{img_filepath.split('image')[0]}/fm_vis/"

        feature_map_path = f"{feature_map_dir}/{img_filepath.split('/')[-1].split('.')[0]}.png"
        os.makedirs(feature_map_dir, exist_ok=True) 
        to_pil = T.ToPILImage()
        img = imgs[0][0][:3]
        img = img.detach().cpu()
        img_min, img_max = img.min(), img.max()
        img = (img - img_min) / (img_max - img_min + 1e-5)
        img = to_pil(img)
        img.save(feature_map_path)


        x = torch.stack(fused_volumes, dim=0)
        x = self.neck_3d(x)

        return x, torch.stack(valid_preds).float()

    def project_volume_to_image(volume, voxel_coords, proj_mat, img_shape):
        """
        Project 3D volume into image space and sample features.

        Args:
            volume (Tensor): (C, Z, Y, X)
            voxel_coords (Tensor): (N, 3)
            proj_mat (Tensor): (4, 4)
            img_shape (tuple): (H, W)

        Returns:
            Tensor: (C, H, W)
        """
        C, Z, Y, X = volume.shape
        device = volume.device

        ones = torch.ones((voxel_coords.shape[0], 1), device=device)
        homo_coords = torch.cat([voxel_coords, ones], dim=1).T
        cam_coords = proj_mat @ homo_coords
        img_coords = cam_coords[:2] / cam_coords[2:].clamp(min=1e-5)
        img_coords = img_coords.T

        H, W = img_shape
        norm_coords = img_coords.clone()
        norm_coords[:, 0] = (norm_coords[:, 0] / (W - 1)) * 2 - 1
        norm_coords[:, 1] = (norm_coords[:, 1] / (H - 1)) * 2 - 1
        norm_coords = norm_coords.view(1, 1, -1, 2)

        volume = volume.view(1, C, Z * Y * X, 1)
        feat_2d = F.grid_sample(volume, norm_coords, align_corners=True).view(C, H, W)
        return feat_2d
    
    def backproject_residual_to_volume(residual_2d, voxel_coords, proj_mat, img_shape, volume_shape):
        """
        Backproject 2D residual into 3D volume.

        Args:
            residual_2d (Tensor): (C, H, W)
            voxel_coords (Tensor): (N, 3)
            proj_mat (Tensor): (4, 4)
            img_shape (tuple): (H, W)
            volume_shape (tuple): (C, Z, Y, X)

        Returns:
            Tensor: (C, Z, Y, X)
        """
        C, Z, Y, X = volume_shape
        device = residual_2d.device

        ones = torch.ones((voxel_coords.shape[0], 1), device=device)
        homo_coords = torch.cat([voxel_coords, ones], dim=1).T
        cam_coords = proj_mat @ homo_coords
        img_coords = cam_coords[:2] / cam_coords[2:].clamp(min=1e-5)
        img_coords = img_coords.T

        H, W = img_shape
        norm_coords = img_coords.clone()
        norm_coords[:, 0] = (norm_coords[:, 0] / (W - 1)) * 2 - 1
        norm_coords[:, 1] = (norm_coords[:, 1] / (H - 1)) * 2 - 1
        norm_coords = norm_coords.view(1, 1, -1, 2)

        residual_2d = residual_2d.view(1, C, H, W)
        sampled_residual = F.grid_sample(residual_2d, norm_coords, align_corners=True)
        sampled_residual = sampled_residual.view(C, Z, Y, X)
        return sampled_residual

    def extract_feat_ART(self, batch_inputs_dict: dict, batch_data_samples: SampleList):
        imgs = batch_inputs_dict['img']
        imgs = torch.stack(imgs, dim=0)
        batch_img_metas = [
            data_samples.metainfo for data_samples in batch_data_samples
        ]

        batch_size, n_views = len(imgs), len(imgs[0])
        C, Z, Y, X = self.neck.out_channels, *self.n_voxels
        
        feats = []
        for i in range(n_views):
            img = imgs[:, i]
            feat = self.backbone(img)
            feats.append(self.neck(feat)[0])

        fused_volumes = []
        for b in range(batch_size):
            img = imgs[b, 0]
            fused_volumes.append(volume = torch.zeros((C, Z, Y, X), device=img.device))

        # iterate over each view (within the batch), each iteration yields 4 sets of points
        iters = 5
        for _ in range(iters):
            for i in range(n_views):
                img = imgs[:, i]
                points = self.prior_generator.grid_anchors([self.n_voxels[::-1]], device=img.device)[0][:, :3]
                feat = feats[i]
                for b in range(batch_size):
                    img_meta = batch_img_metas[b]
                    img_shape =  img_meta['img_shape'][:2]
                    proj_mat = torch.tensor(get_proj_mat_by_coord_type(img_meta, self.coord_type)[i], dtype=points.dtype, device=points.device)
                    simulated_feat = ImVoxelNet.project_volume_to_image(fused_volume, points, proj_mat, img_shape)
                    
                    residual = feat - simulated_feat
                    
                    correction = ImVoxelNet.backproject_residual_to_volume(residual, proj_mat, img_shape, fused_volume.shape)
                    fused_volume += self.alpha * correction

        valid_preds = []
        for volume in fused_volumes:
            valid_preds.append(~torch.all(volume == 0, dim=0, keepdim=True))
        
        x = torch.stack(fused_volumes, dim=0)
        x = self.neck_3d(x)

        return x, torch.stack(valid_preds).float()

    def loss(self, batch_inputs_dict: dict, batch_data_samples: SampleList,
             **kwargs) -> Union[dict, list]:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            batch_inputs_dict (dict): The model input dict which include
                the 'imgs' key.

                    - imgs (torch.Tensor, optional): Image of each sample.
            batch_data_samples (list[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components.
        """
        x, valid_preds = self.extract_feat(batch_inputs_dict,
                                           batch_data_samples)
        # For indoor datasets ImVoxelNet uses ImVoxelHead that handles
        # mask of visible voxels.
        if self.coord_type == 'DEPTH' or self.coord_type == 'CAMERA':
            x += (valid_preds, )
        losses = self.bbox_head.loss(x, batch_data_samples, **kwargs)
        return losses

    def predict(self, batch_inputs_dict: dict, batch_data_samples: SampleList,
                **kwargs) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            batch_inputs_dict (dict): The model input dict which include
                the 'imgs' key.

                    - imgs (torch.Tensor, optional): Image of each sample.

            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance_3d`, `gt_panoptic_seg_3d` and `gt_sem_seg_3d`.

        Returns:
            list[:obj:`Det3DDataSample`]: Detection results of the
            input images. Each Det3DDataSample usually contain
            'pred_instances_3d'. And the ``pred_instances_3d`` usually
            contains following keys.

                - scores_3d (Tensor): Classification scores, has a shape
                    (num_instance, )
                - labels_3d (Tensor): Labels of bboxes, has a shape
                    (num_instances, ).
                - bboxes_3d (Tensor): Contains a tensor with shape
                    (num_instances, C) where C >=7.
        """
        x, valid_preds = self.extract_feat(batch_inputs_dict,
                                           batch_data_samples)
        # For indoor datasets ImVoxelNet uses ImVoxelHead that handles
        # mask of visible voxels.
        if self.coord_type == 'DEPTH' or self.coord_type == 'CAMERA':
            x += (valid_preds, )
        results_list = \
            self.bbox_head.predict(x, batch_data_samples, **kwargs)
        predictions = self.add_pred_to_datasample(batch_data_samples,
                                                  results_list)
        return predictions

    def _forward(self, batch_inputs_dict: dict, batch_data_samples: SampleList,
                 *args, **kwargs) -> Tuple[List[torch.Tensor]]:
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.

        Args:
            batch_inputs_dict (dict): The model input dict which include
                the 'imgs' key.

                    - imgs (torch.Tensor, optional): Image of each sample.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance_3d`, `gt_panoptic_seg_3d` and `gt_sem_seg_3d`.

        Returns:
            tuple[list]: A tuple of features from ``bbox_head`` forward.
        """
        x, valid_preds = self.extract_feat(batch_inputs_dict,
                                           batch_data_samples)
        # For indoor datasets ImVoxelNet uses ImVoxelHead that handles
        # mask of visible voxels.
        if self.coord_type == 'DEPTH' or self.coord_type == 'CAMERA':
            x += (valid_preds, )
        
        results = self.bbox_head.forward(x)
        return results

    def convert_to_datasample(
        self,
        data_samples: SampleList,
        data_instances_3d: OptInstanceList = None,
        data_instances_2d: OptInstanceList = None,
    ) -> SampleList:
        """Convert results list to `Det3DDataSample`.

        Subclasses could override it to be compatible for some multi-modality
        3D detectors.

        Args:
            data_samples (list[:obj:`Det3DDataSample`]): The input data.
            data_instances_3d (list[:obj:`InstanceData`], optional): 3D
                Detection results of each sample.
            data_instances_2d (list[:obj:`InstanceData`], optional): 2D
                Detection results of each sample.

        Returns:
            list[:obj:`Det3DDataSample`]: Detection results of the
            input. Each Det3DDataSample usually contains
            'pred_instances_3d'. And the ``pred_instances_3d`` normally
            contains following keys.

            - scores_3d (Tensor): Classification scores, has a shape
              (num_instance, )
            - labels_3d (Tensor): Labels of 3D bboxes, has a shape
              (num_instances, ).
            - bboxes_3d (Tensor): Contains a tensor with shape
              (num_instances, C) where C >=7.

            When there are image prediction in some models, it should
            contains  `pred_instances`, And the ``pred_instances`` normally
            contains following keys.

            - scores (Tensor): Classification scores of image, has a shape
              (num_instance, )
            - labels (Tensor): Predict Labels of 2D bboxes, has a shape
              (num_instances, ).
            - bboxes (Tensor): Contains a tensor with shape
              (num_instances, 4).
        """

        assert (data_instances_2d is not None) or \
               (data_instances_3d is not None),\
               'please pass at least one type of data_samples'

        if data_instances_2d is None:
            data_instances_2d = [
                InstanceData() for _ in range(len(data_instances_3d))
            ]
        if data_instances_3d is None:
            data_instances_3d = [
                InstanceData() for _ in range(len(data_instances_2d))
            ]

        for i, data_sample in enumerate(data_samples):
            data_sample.pred_instances_3d = data_instances_3d[i]
            data_sample.pred_instances = data_instances_2d[i]
        return data_samples

class SharpeningFilter(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        kernel = torch.tensor([
            [-1, -1, -1],
            [-1, 8, -1],
            [-1, -1, -1]
        ], dtype=torch.float32).view(1, 1, 3, 3).repeat(channels, 1, 1, 1)
        self.register_buffer('kernel', kernel)
        self.groups = channels
    
    def forward(self, x):
        return F.conv2d(x, self.kernel, padding=1, groups=self.groups)