data_root = '/data/data/1000'

_base_ = [
    '../_base_/schedules/mmdet-schedule-1x.py', '../_base_/default_runtime.py'
]
prior_generator = dict(
    type='AlignedAnchor3DRangeGenerator',
    ranges=[[-5, -5, -5, 5, 5, 5]],
    rotations=[.0])
model = dict(
    type='ImVoxelNet',
    backbone=dict(
        type='mmdet.ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50'),
        style='pytorch'),
    neck=dict(
        type='mmdet.FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=4),
    neck_3d=dict(
        type='IndoorImVoxelNeck',
        in_channels=256,
        out_channels=64,
        n_blocks=[1, 1, 1]),
    bbox_head=dict(
        type='ImVoxelHead',
        n_classes=4,
        n_levels=3,
        n_channels=64,
        n_reg_outs=7,
        pts_assign_threshold=3,
        pts_center_threshold=1,
        prior_generator=prior_generator),
    prior_generator=prior_generator,
    n_voxels=[40, 40, 16],
    coord_type='CAMERA',
    train_cfg=dict(),
    test_cfg=dict(nms_pre=1000, iou_thr=.25, score_thr=.01))

dataset_type = 'CustomVisualDataset'
class_names = [
    'Cube'
]
metainfo = dict(CLASSES=class_names)

backend_args = None

train_pipeline = [
    dict(type='LoadAnnotations3D', backend_args=backend_args),
    dict(type='LoadMultiViewImageFromFiles', num_views=2),
    # dict(type='Resize', scale=(640, 480), keep_ratio=True),
    # dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    dict(type='Pack3DDetInputs', keys=['img', 'gt_bboxes_3d', 'gt_labels_3d'])
]
test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles'),
    dict(type='Pack3DDetInputs', keys=['img'])
]

train_dataloader = dict(
    batch_size=4,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='RepeatDataset',
        times=2,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file='custom_visual_infos_train.pkl',
            pipeline=train_pipeline,
            test_mode=False,
            filter_empty_gt=True,
            box_type_3d='Camera',
            metainfo=metainfo,
            backend_args=backend_args)))
val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='custom_visual_infos_val.pkl',
        pipeline=test_pipeline,
        test_mode=True,
        box_type_3d='Camera',
        metainfo=metainfo,
        backend_args=backend_args))
test_dataloader = val_dataloader

val_evaluator = dict(type='IndoorMetric')
test_evaluator = val_evaluator

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(_delete_=True, type='AdamW', lr=0.0001, weight_decay=0.0001),
    paramwise_cfg=dict(custom_keys={'backbone': dict(lr_mult=0.1, decay_mult=1.0)}),
    clip_grad=dict(max_norm=35., norm_type=2))
param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=12,
        by_epoch=True,
        milestones=[8, 11],
        gamma=0.1)
]

vis_backends = [dict(type='LocalVisBackend')]  # Visualization backends. Refer to https://mmengine.readthedocs.io/en/latest/advanced_tutorials/visualization.html
visualizer = dict(type='Det3DLocalVisualizer', vis_backends=vis_backends, name='visualizer')

# hooks
default_hooks = dict(checkpoint=dict(type='CheckpointHook', max_keep_ckpts=1), visualization=dict(type='Det3DVisualizationHook'))

# runtime
find_unused_parameters = True  # only 1 of 4 FPN outputs is used
