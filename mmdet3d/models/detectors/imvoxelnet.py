# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple, Union

import torch
from mmengine.structures import InstanceData

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
        valid_mask = valid_mask[0].cpu().detach().numpy()  # Remove channel dim

        zyx_indices = np.stack(np.nonzero(valid_mask), axis=-1)  # [N, 3] in (z, y, x)

        if len(zyx_indices) == 0:
            print("No valid points to save.")
            return

        # Convert voxel indices to world coordinates
        voxel_origin = np.array(point_cloud_range[:3])  # (x_min, y_min, z_min)
        voxel_size = np.array(voxel_size)

        xyz_points = zyx_indices[:, [2, 1, 0]] * voxel_size + voxel_origin  # Reorder to (x, y, z)

        # Optional: get color/intensity from volume[0] or mean across C channels
        intensity = volume.mean(0)[valid_mask] * 255  # [N]
        intensity = np.clip(intensity, 0, 255).astype(np.uint8)

        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz_points)
        pcd.colors = o3d.utility.Vector3dVector(np.stack([intensity]*3, axis=-1) / 255.0)  # Grayscale

        o3d.io.write_point_cloud(filename, pcd)
        print(f"Saved point cloud to {filename}")


    def extract_feat(self, batch_inputs_dict: dict,
                     batch_data_samples: SampleList):
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
        imgs = batch_inputs_dict['img']
        imgs = torch.stack(imgs, dim=0)
        batch_img_metas = [
            data_samples.metainfo for data_samples in batch_data_samples
        ]

        batch_size, n_views = len(imgs), len(imgs[0])
        all_volumes = [[] for _ in range(batch_size)] # [B, V, ...]
        all_valid_preds = [[] for _ in range(batch_size)]

        # iterate over each view (within the batch), each iteration yields 4 sets of points
        for i in range(len(imgs[0])):
            img = imgs[:, i]
            img = img[:, :3].float()
            x = self.backbone(img)
            x = self.neck(x)[0]
            points = self.prior_generator.grid_anchors([self.n_voxels[::-1]], device=img.device)[0][:, :3]
        
            for b in range(batch_size):
                img_meta = batch_img_metas[b]

                scale = img_meta.get('scale_factor', 1)
                if isinstance(scale, (list, tuple)):
                    img_scale_factor = points.new_tensor(scale[:2])
                else:
                    img_scale_factor = points.new_tensor([scale, scale])

                img_flip = img_meta.get('flip', False)
                img_crop_offset = points.new_tensor(img_meta.get('img_crop_offset', 0))

                proj_mat = points.new_tensor(get_proj_mat_by_coord_type(img_meta, self.coord_type)[i])
                volume = point_sample(
                    img_meta,
                    img_features=x[b][None, ...],
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
                all_volumes[b].append(volume.reshape(self.n_voxels[::-1] + [-1]).permute(3, 2, 1, 0))
                all_valid_preds[b].append(~torch.all(volume == 0, dim=0, keepdim=True))

        fused_volumes = []
        valid_preds = []

        for vols, preds in zip(all_volumes, all_valid_preds):
            vols = torch.stack(vols, dim=0)

            fused_volume = vols.mean(dim=0)
            valid_pred = ~torch.all(fused_volume == 0, dim=0, keepdim=True)

            fused_volumes.append(fused_volume)
            valid_preds.append(valid_pred)

        self.save_pointcloud_from_voxels(
            fused_volumes[0],
            valid_preds[0],
            filename='first_scene.ply'
        )

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
