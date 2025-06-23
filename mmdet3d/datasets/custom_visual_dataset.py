import os.path as osp
from typing import Callable, List, Optional, Union

import numpy as np

from mmdet3d.registry import DATASETS
from mmdet3d.structures import CameraInstance3DBoxes
from .det3d_dataset import Det3DDataset


@DATASETS.register_module()
class CustomVisualDataset(Det3DDataset):

    METAINFO = {
        'classes': ('Cube')
    }

    def __init__(self,
                 data_root: str,
                 ann_file: str,
                 metainfo: Optional[dict] = None,
                 data_prefix: dict = dict(img='images'),
                 pipeline: List[Union[dict, Callable]] = [],
                 default_cam_key: str = 'CAM0',
                 modality: dict = dict(use_camera=True, use_lidar=False),
                 box_type_3d: str = 'Camera',
                 filter_empty_gt: bool = True,
                 test_mode: bool = False,
                 **kwargs) -> None:
        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            metainfo=metainfo,
            data_prefix=data_prefix,
            pipeline=pipeline,
            default_cam_key=default_cam_key,
            modality=modality,
            box_type_3d=box_type_3d,
            filter_empty_gt=filter_empty_gt,
            test_mode=test_mode,
            **kwargs)
        
        assert 'use_camera' in self.modality
        assert self.modality['use_camera'] or self.modality['use_lidar']

    def parse_data_info(self, info: dict) -> dict:
        """Process the raw data info.

        Convert all relative path of needed modality data file to
        the absolute path. And process
        the `instances` field to `ann_info` in training stage.

        Args:
            info (dict): Raw info dict.

        Returns:
            dict: Has `ann_info` in training stage. And
            all path has been converted to absolute path.
        """
        print(self.data_prefix.get('img', ''))

        if self.modality['use_camera']:
            for cam_id, img_info in info['images'].items():
                if 'img_path' in img_info:
                    idx = int(osp.splitext(osp.basename(img_info['img_path']))[0])
                    scene_path = osp.join(self.data_prefix.get('img', ''), f"images_{idx}")
                    img_info['img_path'] = osp.join(scene_path, img_info['img_path'])
            if self.default_cam_key is not None:
                info['img_path'] = info['images'][self.default_cam_key]['img_path']
                info['cam2img'] = np.array(info['images'][self.default_cam_key]['cam2img'], dtype=np.float32)

        if not self.test_mode:
            info['ann_info'] = self.parse_ann_info(info)
        if self.test_mode and self.load_eval_anns:
            info['eval_ann_info'] = self.parse_ann_info(info)

        return info

    def parse_ann_info(self, info: dict) -> dict:
        """Process the `instances` in data info to `ann_info`.

        Args:
            info (dict): Info dict.

        Returns:
            dict: Processed `ann_info`
        """
        ann_info = super().parse_ann_info(info)
        # process data without any annotations
        if ann_info is None:
            ann_info = dict()
            ann_info['gt_bboxes_3d'] = np.zeros((0, 6), dtype=np.float32)
            ann_info['gt_labels_3d'] = np.zeros((0, ), dtype=np.int64)
        # to target box structure
        ann_info['gt_bboxes_3d'] = CameraInstance3DBoxes(ann_info['gt_bboxes_3d']).convert_to(self.box_mode_3d)
        ann_info['gt_bboxes_3d'].rotate(np.array([
            [ 0, 1, 0],
            [1, 0, 0],
            [0, 0, 1]
        ]))

        return ann_info
