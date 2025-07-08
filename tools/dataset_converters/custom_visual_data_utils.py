import os
from concurrent import futures as futures
from os import path as osp

import mmcv
import mmengine
import numpy as np


class CustomVisualInstance(object):
    
    def __init__(self, line):
        data = line.split(' ')
        data[:-1] = [float(x) for x in data[:-1]]
        self.classname = data[-1]
        self.centroid = np.array([data[0], data[1], data[2]])
        self.size = np.array([data[3], data[4], data[5]])
        self.yaw = np.array([data[6]])
        self.length = data[3]
        self.width = data[4]
        self.height = data[5]
        self.heading_angle = self.yaw
        self.box3d = np.concatenate([self.centroid, self.size, self.yaw])

class CustomVisualData(object):

    def __init__(self, root_path, split='train'):
        self.root_dir = root_path
        self.split = split
        self.split_dir = osp.join(root_path, 'ImageSets')
        self.classes = ['Cube']
        self.cat2label = {cat: self.classes.index(cat) for cat in self.classes}
        self.label2cat = {
            label: self.classes[label]
            for label in range(len(self.classes))
        }

        assert split in ['train', 'val']
        split_file = osp.join(self.split_dir, f'{split}.txt')
        mmengine.check_file_exist(split_file)
        self.sample_id_list = map(int, mmengine.list_from_file(split_file))
        self.image_dir = osp.join(self.root_dir, 'images')
        self.calib_dir = osp.join(self.root_dir, 'calib')
        self.label_dir = osp.join(self.root_dir, 'labels')
        calib_filepath = osp.join(self.calib_dir, f'000001.txt')
        lines = [line.rstrip() for line in open(calib_filepath)]
        n_cams = int(len(lines) / 2)
        self.cams = []
        for i in range(n_cams+1):
            self.cams.append(f'CAM{i}')
        
    def __len__(self):
        return len(self.sample_id_list)
    
    def get_image(self, idx, cam):
        cam_dir = osp.join(self.image_dir, f'images_{cam}')
        image_filename = osp.join(cam_dir, f'{idx:06d}.png')
        return mmcv.imread(image_filename)
    
    def get_image_shape(self, idx, cam):
        image = self.get_image(idx, cam)
        return np.array(image.shape[:2], dtype=np.int32)
    
    def get_calibration(self, idx):
        calib_filepath = osp.join(self.calib_dir, f'{idx:06d}.txt')
        lines = [line.rstrip() for line in open(calib_filepath)]
        
        calibs = []
        print(self.get_n_cams())
        for i in range(0, self.get_n_cams() * 2, 2):
            K = np.array([float(x) for x in lines[i].split(' ')])
            K = np.reshape(K, (3, 3), order='F').astype(np.float32)
            Rt = np.array([float(x) for x in lines[i+1].split(' ')])
            Rt = np.reshape(Rt, (4, 4), order='F').astype(np.float32)
            calibs.append((K, Rt))

        return calibs
    
    def get_label_objects(self, idx):
        label_filename = osp.join(self.label_dir, f'{idx:06d}.txt')
        lines = [line.rstrip() for line in open(label_filename)]
        objects = [CustomVisualInstance(line) for line in lines]
        return objects
    
    def get_n_cams(self):
        return len(self.cams)
    
    def get_infos(self, num_workers=4, has_label=True, sample_id_list=None):

        def process_single_scene(sample_idx):
            print(f'{self.split} sample_idx: {sample_idx}')
            info = dict()
            info['images'] = {}
            info['calib'] = {}
            

            calibs = self.get_calibration(sample_idx)
            for i in range(1, self.get_n_cams() + 1):
                cam_path = osp.join(self.image_dir, f'images_{i}')
                image_path = osp.join(cam_path, f'{sample_idx:06d}.png')
                image_info = {
                    'image_idx': sample_idx,
                    'image_shape': self.get_image_shape(sample_idx, i),
                    'image_path': image_path
                }
                info['images'][f'CAM{i-1}'] = image_info
                K, Rt = calibs[i-1]
                calib_info = {'K': K, 'Rt': Rt}
                info['calib'][f'CAM{i-1}'] = calib_info

            if has_label:
                obj_list = self.get_label_objects(sample_idx)
                annotations = {}
                annotations['gt_num'] = len([
                    obj.classname for obj in obj_list
                    if obj.classname in self.cat2label.keys()
                ])
                if annotations['gt_num'] != 0:
                    annotations['name'] = np.array([
                        obj.classname for obj in obj_list
                        if obj.classname in self.cat2label.keys()
                    ])

                    # annotations['bbox'] = np.concatenate([
                    #     obj.box2d.reshape(1, 4) for obj in obj_list
                    #     if obj.classname in self.cat2label.keys()
                    # ], axis=0)

                    annotations['location'] = np.concatenate([
                        obj.centroid.reshape(1, 3) for obj in obj_list
                        if obj.classname in self.cat2label.keys()
                    ], axis=0)

                    annotations['dimensions'] = np.array([
                        [obj.length, obj.width, obj.height] for obj in obj_list
                        if obj.classname in self.cat2label.keys()
                    ])  # lwh (depth) format

                    annotations['rotation_y'] = np.array([
                        obj.heading_angle for obj in obj_list
                        if obj.classname in self.cat2label.keys()
                    ])

                    annotations['index'] = np.arange(len(obj_list), dtype=np.int32)
                    annotations['class'] = np.array([
                        self.cat2label[obj.classname] for obj in obj_list
                        if obj.classname in self.cat2label.keys()
                    ])

                    annotations['gt_boxes_upright_depth'] = np.stack([
                        obj.box3d for obj in obj_list
                        if obj.classname in self.cat2label.keys()
                    ],axis=0)  # (K,8)
                    
                info['annos'] = annotations
            return info
        
        sample_id_list = sample_id_list if \
        sample_id_list is not None else self.sample_id_list
        with futures.ThreadPoolExecutor(num_workers) as executor:
            infos = executor.map(process_single_scene, sample_id_list)
        return list(infos)