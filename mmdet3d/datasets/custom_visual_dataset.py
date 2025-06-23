import os.path as osp
from typing import Callable, List, Optional, Union

import numpy as np

from mmdet3d.registry import DATASETS
from mmdet3d.structures import CameraInstance3DBoxes
from .det3d_dataset import Det3DDataset
from mmengine.fileio import join_path, list_from_file, load

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
        ann_info['gt_bboxes_3d'] = CameraInstance3DBoxes(ann_info['gt_bboxes_3d'], origin=(0.5, 0.5, 0.5)).convert_to(self.box_mode_3d)
        # ann_info['gt_bboxes_3d'].rotate(np.array([
        #     [ 1, 0, 0],
        #     [0, 0, -1],
        #     [0, -1, 0]
        # ]))

        return ann_info

    def load_data_list(self) -> List[dict]:
            """Load annotations from an annotation file named as ``self.ann_file``

            If the annotation file does not follow `OpenMMLab 2.0 format dataset
            <https://mmengine.readthedocs.io/en/latest/advanced_tutorials/basedataset.html>`_ .
            The subclass must override this method for load annotations. The meta
            information of annotation file will be overwritten :attr:`METAINFO`
            and ``metainfo`` argument of constructor.

            Returns:
                list[dict]: A list of annotation.
            """  # noqa: E501
            # `self.ann_file` denotes the absolute annotation file path if
            # `self.root=None` or relative path if `self.root=/path/to/data/`.
            annotations = load(self.ann_file)
            print(self.ann_file)
            print(annotations)
            if not isinstance(annotations, dict):
                raise TypeError(f'The annotations loaded from annotation file '
                                f'should be a dict, but got {type(annotations)}!')
            if 'data_list' not in annotations or 'metainfo' not in annotations:
                raise ValueError('Annotation must have data_list and metainfo '
                                'keys')
            metainfo = annotations['metainfo']
            raw_data_list = annotations['data_list']

            # Meta information load from annotation file will not influence the
            # existed meta information load from `BaseDataset.METAINFO` and
            # `metainfo` arguments defined in constructor.
            for k, v in metainfo.items():
                self._metainfo.setdefault(k, v)

            # load and parse data_infos.
            data_list = []
            for raw_data_info in raw_data_list:
                # parse raw data information to target format
                data_info = self.parse_data_info(raw_data_info)
                if isinstance(data_info, dict):
                    # For image tasks, `data_info` should information if single
                    # image, such as dict(img_path='xxx', width=360, ...)
                    data_list.append(data_info)
                elif isinstance(data_info, list):
                    # For video tasks, `data_info` could contain image
                    # information of multiple frames, such as
                    # [dict(video_path='xxx', timestamps=...),
                    #  dict(video_path='xxx', timestamps=...)]
                    for item in data_info:
                        if not isinstance(item, dict):
                            raise TypeError('data_info must be list of dict, but '
                                            f'got {type(item)}')
                    data_list.extend(data_info)
                else:
                    raise TypeError('data_info should be a dict or list of dict, '
                                    f'but got {type(data_info)}')

            return data_list