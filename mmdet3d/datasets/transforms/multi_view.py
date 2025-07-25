import numpy as np

from mmdet3d.registry import TRANSFORMS
from mmengine.dataset import Compose

@TRANSFORMS.register_module()
class MultiViewPipeline:
    def __init__(self, transforms, n_images):
        self.transforms = Compose(transforms)
        self.n_images = n_images

    def __call__(self, results):
        imgs = []
        extrinsics = []
        print(results.keys())
        ids = np.arange(len(results['images']))
        replace = True if self.n_images > len(ids) else False
        ids = np.random.choice(ids, self.n_images, replace=replace)
        for i in range(self.n_images):
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