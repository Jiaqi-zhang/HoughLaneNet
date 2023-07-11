dataset_type = 'LLAMASDataset'
data_root = './datasets/LLAMAS/'
work_dir = './work_dirs/final_newcon/llamas/res_large'
img_norm_cfg = dict(
    mean=[75.3, 76.6, 77.6], std=[50.5, 53.8, 54.3], to_rgb=True)
img_scale = (640, 360)
ori_scale = (1276, 717)
line_width = 5
max_num_lane = 4
hough_point_radius = 3
hough_point_ratio = 1.0
total_epochs = 20
batch_size = 2
threshold = 0.1
nms_kernel_size = 5
select_mode = 'nms'
line_mode = 'line'
d_model = 192
d_houghs = None
d_ins = 48
groups = 4
hough_scales = [3, 3, 3]
num_angle = 360
num_rho = 360
fea_size = (90, 160)
train_compose = dict(bboxes=False, keypoints=True, masks=False)
train_al_pipeline = [
    dict(
        type='Compose', params=dict(bboxes=False, keypoints=True,
                                    masks=False)),
    dict(type='Resize', height=360, width=640, p=1),
    dict(
        type='OneOf',
        transforms=[
            dict(
                type='RGBShift',
                r_shift_limit=15,
                g_shift_limit=15,
                b_shift_limit=15,
                p=1.0),
            dict(
                type='HueSaturationValue',
                hue_shift_limit=(-10, 10),
                sat_shift_limit=(-15, 15),
                val_shift_limit=(-10, 10),
                p=1.0),
            dict(type='FancyPCA', alpha=0.2, p=1.0)
        ],
        p=0.6),
    dict(type='ImageCompression', quality_lower=85, quality_upper=95, p=0.5),
    dict(type='CLAHE', p=0.5),
    dict(
        type='OneOf',
        transforms=[
            dict(type='Blur', blur_limit=3, p=1.0),
            dict(type='MedianBlur', blur_limit=3, p=1.0),
            dict(type='GaussianBlur', blur_limit=(3, 5), p=1.0)
        ],
        p=0.1),
    dict(
        type='ColorJitter',
        brightness=0.5,
        contrast=0.5,
        saturation=0.5,
        hue=0.5,
        always_apply=True,
        p=1),
    dict(
        type='RandomBrightnessContrast',
        brightness_limit=0.25,
        contrast_limit=0.25,
        p=0.6),
    dict(
        type='OneOf',
        transforms=[
            dict(
                type='RandomFog',
                fog_coef_lower=0.1,
                fog_coef_upper=0.3,
                p=1.0),
            dict(type='RandomShadow', p=1.0)
        ],
        p=0.1),
    dict(type='HorizontalFlip', p=0.5),
    dict(
        type='ShiftScaleRotate',
        shift_limit=0.12,
        scale_limit=0.08,
        rotate_limit=25,
        p=0.8)
]
test_al_pipeline = [
    dict(
        type='Compose', params=dict(bboxes=False, keypoints=True,
                                    masks=False)),
    dict(type='Resize', height=360, width=640, p=1)
]
train_pipeline = [
    dict(
        type='albumentation',
        pipelines=[
            dict(
                type='Compose',
                params=dict(bboxes=False, keypoints=True, masks=False)),
            dict(type='Resize', height=360, width=640, p=1),
            dict(
                type='OneOf',
                transforms=[
                    dict(
                        type='RGBShift',
                        r_shift_limit=15,
                        g_shift_limit=15,
                        b_shift_limit=15,
                        p=1.0),
                    dict(
                        type='HueSaturationValue',
                        hue_shift_limit=(-10, 10),
                        sat_shift_limit=(-15, 15),
                        val_shift_limit=(-10, 10),
                        p=1.0),
                    dict(type='FancyPCA', alpha=0.2, p=1.0)
                ],
                p=0.6),
            dict(
                type='ImageCompression',
                quality_lower=85,
                quality_upper=95,
                p=0.5),
            dict(type='CLAHE', p=0.5),
            dict(
                type='OneOf',
                transforms=[
                    dict(type='Blur', blur_limit=3, p=1.0),
                    dict(type='MedianBlur', blur_limit=3, p=1.0),
                    dict(type='GaussianBlur', blur_limit=(3, 5), p=1.0)
                ],
                p=0.1),
            dict(
                type='ColorJitter',
                brightness=0.5,
                contrast=0.5,
                saturation=0.5,
                hue=0.5,
                always_apply=True,
                p=1),
            dict(
                type='RandomBrightnessContrast',
                brightness_limit=0.25,
                contrast_limit=0.25,
                p=0.6),
            dict(
                type='OneOf',
                transforms=[
                    dict(
                        type='RandomFog',
                        fog_coef_lower=0.1,
                        fog_coef_upper=0.3,
                        p=1.0),
                    dict(type='RandomShadow', p=1.0)
                ],
                p=0.1),
            dict(type='HorizontalFlip', p=0.5),
            dict(
                type='ShiftScaleRotate',
                shift_limit=0.12,
                scale_limit=0.08,
                rotate_limit=25,
                p=0.8)
        ]),
    dict(
        type='Normalize',
        mean=[75.3, 76.6, 77.6],
        std=[50.5, 53.8, 54.3],
        to_rgb=True),
    dict(type='DefaultFormatBundle'),
    dict(
        type='CollectLane',
        line_width=5,
        line_mode='line',
        max_num_lane=4,
        num_angle=360,
        num_rho=360,
        hough_point_radius=3,
        hough_point_ratio=1.0,
        keys=['img', 'segment_map', 'hough_map', 'line_map', 'point_list'],
        meta_keys=[
            'filename', 'sub_img_name', 'ori_shape', 'img_shape',
            'img_norm_cfg'
        ])
]
val_pipeline = [
    dict(
        type='albumentation',
        pipelines=[
            dict(
                type='Compose',
                params=dict(bboxes=False, keypoints=True, masks=False)),
            dict(type='Resize', height=360, width=640, p=1),
            dict(
                type='OneOf',
                transforms=[
                    dict(
                        type='RGBShift',
                        r_shift_limit=15,
                        g_shift_limit=15,
                        b_shift_limit=15,
                        p=1.0),
                    dict(
                        type='HueSaturationValue',
                        hue_shift_limit=(-10, 10),
                        sat_shift_limit=(-15, 15),
                        val_shift_limit=(-10, 10),
                        p=1.0),
                    dict(type='FancyPCA', alpha=0.2, p=1.0)
                ],
                p=0.6),
            dict(
                type='ImageCompression',
                quality_lower=85,
                quality_upper=95,
                p=0.5),
            dict(type='CLAHE', p=0.5),
            dict(
                type='OneOf',
                transforms=[
                    dict(type='Blur', blur_limit=3, p=1.0),
                    dict(type='MedianBlur', blur_limit=3, p=1.0),
                    dict(type='GaussianBlur', blur_limit=(3, 5), p=1.0)
                ],
                p=0.1),
            dict(
                type='ColorJitter',
                brightness=0.5,
                contrast=0.5,
                saturation=0.5,
                hue=0.5,
                always_apply=True,
                p=1),
            dict(
                type='RandomBrightnessContrast',
                brightness_limit=0.25,
                contrast_limit=0.25,
                p=0.6),
            dict(
                type='OneOf',
                transforms=[
                    dict(
                        type='RandomFog',
                        fog_coef_lower=0.1,
                        fog_coef_upper=0.3,
                        p=1.0),
                    dict(type='RandomShadow', p=1.0)
                ],
                p=0.1),
            dict(type='HorizontalFlip', p=0.5),
            dict(
                type='ShiftScaleRotate',
                shift_limit=0.12,
                scale_limit=0.08,
                rotate_limit=25,
                p=0.8)
        ]),
    dict(
        type='Normalize',
        mean=[75.3, 76.6, 77.6],
        std=[50.5, 53.8, 54.3],
        to_rgb=True),
    dict(type='DefaultFormatBundle'),
    dict(
        type='CollectLane',
        line_width=5,
        line_mode='line',
        max_num_lane=4,
        num_angle=360,
        num_rho=360,
        hough_point_radius=3,
        hough_point_ratio=1.0,
        keys=['img', 'segment_map', 'hough_map', 'line_map', 'point_list'],
        meta_keys=[
            'filename', 'sub_img_name', 'ori_shape', 'img_shape',
            'img_norm_cfg'
        ])
]
test_pipeline = [
    dict(
        type='albumentation',
        pipelines=[
            dict(
                type='Compose',
                params=dict(bboxes=False, keypoints=True, masks=False)),
            dict(type='Resize', height=360, width=640, p=1)
        ]),
    dict(
        type='Normalize',
        mean=[75.3, 76.6, 77.6],
        std=[50.5, 53.8, 54.3],
        to_rgb=True),
    dict(type='DefaultFormatBundle'),
    dict(
        type='CollectLane',
        line_width=5,
        line_mode='line',
        max_num_lane=4,
        num_angle=360,
        num_rho=360,
        hough_point_radius=3,
        hough_point_ratio=1.0,
        keys=['img', 'segment_map', 'hough_map', 'line_map', 'point_list'],
        meta_keys=[
            'filename', 'sub_img_name', 'ori_shape', 'img_shape',
            'img_norm_cfg'
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type='LLAMASDataset',
        mode='train',
        data_root=data_root,
        data_json_dir='json_lanes_train',
        data_list=[data_root + 'train.txt'],
        pipeline=[
            dict(
                type='albumentation',
                pipelines=[
                    dict(
                        type='Compose',
                        params=dict(bboxes=False, keypoints=True,
                                    masks=False)),
                    dict(type='Resize', height=360, width=640, p=1),
                    dict(
                        type='OneOf',
                        transforms=[
                            dict(
                                type='RGBShift',
                                r_shift_limit=15,
                                g_shift_limit=15,
                                b_shift_limit=15,
                                p=1.0),
                            dict(
                                type='HueSaturationValue',
                                hue_shift_limit=(-10, 10),
                                sat_shift_limit=(-15, 15),
                                val_shift_limit=(-10, 10),
                                p=1.0),
                            dict(type='FancyPCA', alpha=0.2, p=1.0)
                        ],
                        p=0.6),
                    dict(
                        type='ImageCompression',
                        quality_lower=85,
                        quality_upper=95,
                        p=0.5),
                    dict(type='CLAHE', p=0.5),
                    dict(
                        type='OneOf',
                        transforms=[
                            dict(type='Blur', blur_limit=3, p=1.0),
                            dict(type='MedianBlur', blur_limit=3, p=1.0),
                            dict(
                                type='GaussianBlur', blur_limit=(3, 5), p=1.0)
                        ],
                        p=0.1),
                    dict(
                        type='ColorJitter',
                        brightness=0.5,
                        contrast=0.5,
                        saturation=0.5,
                        hue=0.5,
                        always_apply=True,
                        p=1),
                    dict(
                        type='RandomBrightnessContrast',
                        brightness_limit=0.25,
                        contrast_limit=0.25,
                        p=0.6),
                    dict(
                        type='OneOf',
                        transforms=[
                            dict(
                                type='RandomFog',
                                fog_coef_lower=0.1,
                                fog_coef_upper=0.3,
                                p=1.0),
                            dict(type='RandomShadow', p=1.0)
                        ],
                        p=0.1),
                    dict(type='HorizontalFlip', p=0.5),
                    dict(
                        type='ShiftScaleRotate',
                        shift_limit=0.12,
                        scale_limit=0.08,
                        rotate_limit=25,
                        p=0.8)
                ]),
            dict(
                type='Normalize',
                mean=[75.3, 76.6, 77.6],
                std=[50.5, 53.8, 54.3],
                to_rgb=True),
            dict(type='DefaultFormatBundle'),
            dict(
                type='CollectLane',
                line_width=5,
                line_mode='line',
                max_num_lane=4,
                num_angle=360,
                num_rho=360,
                hough_point_radius=3,
                hough_point_ratio=1.0,
                keys=[
                    'img', 'segment_map', 'hough_map', 'line_map', 'point_list'
                ],
                meta_keys=[
                    'filename', 'sub_img_name', 'ori_shape', 'img_shape',
                    'img_norm_cfg'
                ])
        ],
        test_mode=False,
        ori_scale=(1276, 717),
        img_scale=(640, 360),
        max_num_lane=4),
    val=dict(
        type='LLAMASDataset',
        mode='valid',
        data_root=data_root,
        data_json_dir='json_lanes_valid',
        data_list=[data_root + 'valid.txt'],
        pipeline=[
            dict(
                type='albumentation',
                pipelines=[
                    dict(
                        type='Compose',
                        params=dict(bboxes=False, keypoints=True,
                                    masks=False)),
                    dict(type='Resize', height=360, width=640, p=1),
                    dict(
                        type='OneOf',
                        transforms=[
                            dict(
                                type='RGBShift',
                                r_shift_limit=15,
                                g_shift_limit=15,
                                b_shift_limit=15,
                                p=1.0),
                            dict(
                                type='HueSaturationValue',
                                hue_shift_limit=(-10, 10),
                                sat_shift_limit=(-15, 15),
                                val_shift_limit=(-10, 10),
                                p=1.0),
                            dict(type='FancyPCA', alpha=0.2, p=1.0)
                        ],
                        p=0.6),
                    dict(
                        type='ImageCompression',
                        quality_lower=85,
                        quality_upper=95,
                        p=0.5),
                    dict(type='CLAHE', p=0.5),
                    dict(
                        type='OneOf',
                        transforms=[
                            dict(type='Blur', blur_limit=3, p=1.0),
                            dict(type='MedianBlur', blur_limit=3, p=1.0),
                            dict(
                                type='GaussianBlur', blur_limit=(3, 5), p=1.0)
                        ],
                        p=0.1),
                    dict(
                        type='ColorJitter',
                        brightness=0.5,
                        contrast=0.5,
                        saturation=0.5,
                        hue=0.5,
                        always_apply=True,
                        p=1),
                    dict(
                        type='RandomBrightnessContrast',
                        brightness_limit=0.25,
                        contrast_limit=0.25,
                        p=0.6),
                    dict(
                        type='OneOf',
                        transforms=[
                            dict(
                                type='RandomFog',
                                fog_coef_lower=0.1,
                                fog_coef_upper=0.3,
                                p=1.0),
                            dict(type='RandomShadow', p=1.0)
                        ],
                        p=0.1),
                    dict(type='HorizontalFlip', p=0.5),
                    dict(
                        type='ShiftScaleRotate',
                        shift_limit=0.12,
                        scale_limit=0.08,
                        rotate_limit=25,
                        p=0.8)
                ]),
            dict(
                type='Normalize',
                mean=[75.3, 76.6, 77.6],
                std=[50.5, 53.8, 54.3],
                to_rgb=True),
            dict(type='DefaultFormatBundle'),
            dict(
                type='CollectLane',
                line_width=5,
                line_mode='line',
                max_num_lane=4,
                num_angle=360,
                num_rho=360,
                hough_point_radius=3,
                hough_point_ratio=1.0,
                keys=[
                    'img', 'segment_map', 'hough_map', 'line_map', 'point_list'
                ],
                meta_keys=[
                    'filename', 'sub_img_name', 'ori_shape', 'img_shape',
                    'img_norm_cfg'
                ])
        ],
        test_mode=False,
        ori_scale=(1276, 717),
        img_scale=(640, 360),
        max_num_lane=4,
        cp_work_dir=work_dir + '/outputs',
        samples_per_gpu=1),
    test=dict(
        type='LLAMASDataset',
        mode='test',
        data_root=data_root,
        data_json_dir='json_lanes_valid',
        data_list=[data_root + 'valid.txt'],
        test_suffix='.jpg',
        pipeline=[
            dict(
                type='albumentation',
                pipelines=[
                    dict(
                        type='Compose',
                        params=dict(bboxes=False, keypoints=True,
                                    masks=False)),
                    dict(type='Resize', height=360, width=640, p=1)
                ]),
            dict(
                type='Normalize',
                mean=[75.3, 76.6, 77.6],
                std=[50.5, 53.8, 54.3],
                to_rgb=True),
            dict(type='DefaultFormatBundle'),
            dict(
                type='CollectLane',
                line_width=5,
                line_mode='line',
                max_num_lane=4,
                num_angle=360,
                num_rho=360,
                hough_point_radius=3,
                hough_point_ratio=1.0,
                keys=[
                    'img', 'segment_map', 'hough_map', 'line_map', 'point_list'
                ],
                meta_keys=[
                    'filename', 'sub_img_name', 'ori_shape', 'img_shape',
                    'img_norm_cfg'
                ])
        ],
        test_mode=True,
        ori_scale=(1276, 717),
        img_scale=(640, 360),
        max_num_lane=4,
        cp_work_dir=work_dir + '/outputs',
        samples_per_gpu=16))
model = dict(
    type='LaneDetector',
    base_data=dict(
        image_height=360,
        image_width=640,
        patch_height=10,
        patch_width=10,
        d_model=192),
    backbone=dict(
        type='ResNet',
        depth=101,
        strides=(1, 2, 2, 2),
        num_stages=4,
        out_indices=[0, 1, 2, 3],
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained',
                      checkpoint='torchvision://resnet101')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=192,
        num_outs=4),
    head=dict(
        type='DenseLaneHead',
        image_size=(360, 640),
        max_num_lane=4,
        d_model=192,
        d_ins=48,
        d_houghs=None,
        groups=4,
        hough_scales=[3, 3, 3],
        num_angle=360,
        num_rho=360,
        fea_size=(90, 160),
        threshold=threshold,
        nms_kernel_size=5,
        select_mode='nms'),
    loss_weights=dict(
        seg_weight=100.0,
        hough_weight=1000.0,
        line_weight=100.0,
        range_weight=10.0,
        lane_weight=100.0,
        pos_weight=[10, 10, 10, 10]))
optimizer = dict(
    type='AdamW', lr=0.0003, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-07)
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='step',
    warmup='constant',
    warmup_iters=10,
    warmup_ratio=0.3333333333333333,
    step=1,
    gamma=0.9,
    min_lr=0.0001)
runner = dict(type='EpochBasedRunner', max_epochs=20)
checkpoint_config = dict(interval=1)
log_config = dict(
    interval=20,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])
evaluation = dict(interval=1)
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
find_unused_parameters = True
cudnn_benchmark = True
workflow = [('train', 1), ('val', 1)]
gpu_ids = range(0, 3)
