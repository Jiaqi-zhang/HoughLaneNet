dataset_type = 'TuSimpleDataset'
data_root = './datasets/TuSimple/'
work_dir = './work_dirs/final_smean/tusimple/res_small'
img_norm_cfg = dict(
    mean=[75.3, 76.6, 77.6], std=[50.5, 53.8, 54.3], to_rgb=True)
img_scale = (640, 360)
ori_scale = (1280, 720)
h_samples = [
    160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300,
    310, 320, 330, 340, 350, 360, 370, 380, 390, 400, 410, 420, 430, 440, 450,
    460, 470, 480, 490, 500, 510, 520, 530, 540, 550, 560, 570, 580, 590, 600,
    610, 620, 630, 640, 650, 660, 670, 680, 690, 700, 710
]
line_width = 5
max_num_lane = 5
hough_point_radius = 3
hough_point_ratio = 1.0
total_epochs = 150
batch_size = 3
threshold = 0.1
select_mode = 'nms'  # [nms, region]
line_mode = 'line'  # [line, hough]

d_model = 128
d_ins = 32
groups = 4
hough_scale = 3
num_angle = 240
num_rho = 240
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
        line_mode=line_mode,
        max_num_lane=5,
        num_angle=240,
        num_rho=240,
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
        line_mode=line_mode,
        max_num_lane=5,
        num_angle=240,
        num_rho=240,
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
        line_mode=line_mode,
        max_num_lane=5,
        num_angle=240,
        num_rho=240,
        hough_point_radius=3,
        hough_point_ratio=1.0,
        keys=['img', 'segment_map', 'hough_map', 'line_map', 'point_list'],
        meta_keys=[
            'filename', 'sub_img_name', 'ori_shape', 'img_shape',
            'img_norm_cfg'
        ])
]
data = dict(
    samples_per_gpu=3,
    workers_per_gpu=3,
    train=dict(
        type='TuSimpleDataset',
        data_root=data_root,
        data_list=[
            data_root + 'label_data_0313.json',
            data_root + 'label_data_0531.json',
            data_root + 'label_data_0601.json',
        ],
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
                line_mode=line_mode,
                max_num_lane=5,
                num_angle=240,
                num_rho=240,
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
        ori_scale=(1280, 720),
        img_scale=(640, 360),
        max_num_lane=5,
        h_samples=[
            160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260, 270, 280,
            290, 300, 310, 320, 330, 340, 350, 360, 370, 380, 390, 400, 410,
            420, 430, 440, 450, 460, 470, 480, 490, 500, 510, 520, 530, 540,
            550, 560, 570, 580, 590, 600, 610, 620, 630, 640, 650, 660, 670,
            680, 690, 700, 710
        ]),
    val=dict(
        type='TuSimpleDataset',
        data_root=data_root,
        data_list=[data_root + 'test_label.json'],
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
                line_mode=line_mode,
                max_num_lane=5,
                num_angle=240,
                num_rho=240,
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
        ori_scale=(1280, 720),
        img_scale=(640, 360),
        max_num_lane=5,
        h_samples=[
            160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260, 270, 280,
            290, 300, 310, 320, 330, 340, 350, 360, 370, 380, 390, 400, 410,
            420, 430, 440, 450, 460, 470, 480, 490, 500, 510, 520, 530, 540,
            550, 560, 570, 580, 590, 600, 610, 620, 630, 640, 650, 660, 670,
            680, 690, 700, 710
        ],
        cp_work_dir=work_dir + '/outputs',
        samples_per_gpu=1),
    test=dict(
        type='TuSimpleDataset',
        data_root=data_root,
        data_list=[data_root + 'test_label.json'],
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
                line_mode=line_mode,
                max_num_lane=5,
                num_angle=240,
                num_rho=240,
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
        ori_scale=(1280, 720),
        img_scale=(640, 360),
        max_num_lane=5,
        h_samples=[
            160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260, 270, 280,
            290, 300, 310, 320, 330, 340, 350, 360, 370, 380, 390, 400, 410,
            420, 430, 440, 450, 460, 470, 480, 490, 500, 510, 520, 530, 540,
            550, 560, 570, 580, 590, 600, 610, 620, 630, 640, 650, 660, 670,
            680, 690, 700, 710
        ],
        cp_work_dir=work_dir + '/outputs',
        samples_per_gpu=8))
model = dict(
    type='LaneDetector',
    base_data=dict(
        image_height=360,
        image_width=640,
        patch_height=10,
        patch_width=10,
        d_model=128),
    backbone=dict(
        type='ResNet',
        depth=18,
        strides=(1, 2, 2, 2),
        num_stages=4,
        out_indices=[0, 1, 2, 3],
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet18')),
    neck=dict(
        type='FPN',
        in_channels=[64, 128, 256, 512],
        out_channels=128,
        num_outs=4),
    head=dict(
        type='DenseLaneHead',
        image_size=(360, 640),
        max_num_lane=5,
        d_model=128,
        d_ins=32,
        groups=4,
        hough_scale=3,
        num_angle=240,
        num_rho=240,
        fea_size=(90, 160),
        threshold=threshold,
        select_mode=select_mode,
        ),
    loss_weights=dict(
        seg_weight=100.0,
        hough_weight=1000.0,
        line_weight=100.0,
        range_weight=10.0,
        lane_weight=100.0,
        pos_weight=[10, 10, 10, 10, 10]))
optimizer = dict(
    type='AdamW', lr=0.0003, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-07)
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='step',
    warmup='constant',
    warmup_iters=10,
    warmup_ratio=0.3333333333333333,
    step=15,
    gamma=0.9,
    min_lr=0.0001)
runner = dict(type='EpochBasedRunner', max_epochs=150)
checkpoint_config = dict(interval=5)
log_config = dict(
    interval=20,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])
evaluation = dict(interval=5)
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
find_unused_parameters = True
cudnn_benchmark = True
workflow = [('train', 300), ('val', 1)]
