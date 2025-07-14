import os.path as osp
from typing import Callable, List, Optional, Union

import numpy as np

from mmdet3d.registry import DATASETS
from mmdet3d.structures import CameraInstance3DBoxes, DepthInstance3DBoxes
from .det3d_dataset import Det3DDataset
from mmengine.fileio import join_path, list_from_file, load
from torch.utils.data import Dataset
from collections.abc import Mapping
from mmengine.config import Config
import copy
import logging

@DATASETS.register_module()
class CustomVisualDataset(Det3DDataset):

    METAINFO = {
        'classes': ['Cube']
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

        if self.modality['use_camera']:
            for cam_id, img_info in info['images'].items():
                if 'img_path' in img_info:
                    cam_prefix = f"{self.data_prefix['img']}/images_{int(cam_id[-1]) + 1}"
                    img_info['img_path'] = osp.join(cam_prefix, img_info['img_path'])

            # if self.default_cam_key is not None:
            #     info['img_path'] = info['images'][self.default_cam_key]['img_path']
            #     info['cam2img'] = np.array(info['images'][self.default_cam_key]['cam2img'], dtype=np.float32)

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
        ann_info['gt_bboxes_3d'] = CameraInstance3DBoxes(ann_info['gt_bboxes_3d'], origin=(0.5, 0.5, 0.5))

        return ann_info
    
    def _load_metainfo(cls, metainfo: Union[Mapping, Config, None] = None) -> dict:
        """Collect meta information from the dictionary of meta.

        Args:
            metainfo (Mapping or Config, optional): Meta information dict.
                If ``metainfo`` contains existed filename, it will be
                parsed by ``list_from_file``.

        Returns:
            dict: Parsed meta information.
        """
        # avoid `cls.METAINFO` being overwritten by `metainfo`
        cls_metainfo = copy.deepcopy(cls.METAINFO)

        if metainfo is None:
            return cls_metainfo
        if not isinstance(metainfo, (Mapping, Config)):
            raise TypeError('metainfo should be a Mapping or Config, '
                            f'but got {type(metainfo)}')

        for k, v in metainfo.items():
            if isinstance(v, str):
                # If type of value is string, and can be loaded from
                # corresponding backend. it means the file name of meta file.
                try:
                    cls_metainfo[k] = list_from_file(v)
                except (TypeError, FileNotFoundError):
                    print_log(
                        f'{v} is not a meta file, simply parsed as meta '
                        'information',
                        logger='current',
                        level=logging.WARNING)
                    cls_metainfo[k] = v
            else:
                cls_metainfo[k] = v
        return cls_metainfo