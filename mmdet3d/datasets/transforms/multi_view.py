import numpy as np

from mmdet.registry import TRANSFORMS
from mmdet.datasets.transforms import Compose, RandomFlip, LoadImageFromFile


@PIPELINES.register_module()
class MultiViewPipeline:
    def __init__(self, transforms, n_images):
        self.transforms = Compose(transforms)
        self.n_images = n_images

    def __call__(self, results):
        imgs = []
        extrinsics = []
        ids = np.arange(len(results['img_info']))
        replace = True if self.n_images > len(ids) else False
        ids = np.random.choice(ids, self.n_images, replace=replace)
        for i in ids.tolist():
            _results = dict()
            for key in ['img_prefix', 'img_info']:
                _results[key] = results[key][i]
            _results = self.transforms(_results)
            imgs.append(_results['img'])
            extrinsics.append(results['cam2img']['extrinsic'][i])
        for key in _results.keys():
            if key not in ['img', 'img_prefix', 'img_info']:
                results[key] = _results[key]
        results['img'] = imgs
        results['cam2img']['extrinsic'] = extrinsics
        return results