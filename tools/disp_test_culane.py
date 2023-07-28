import argparse
import os
import os.path as osp
from os.path import join as pjoin
import pickle
import time
import warnings
import ujson as json
import cv2
import numpy as np
from tqdm import tqdm
from scipy.interpolate import splprep, splev

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel
from mmcv.runner import (get_dist_info, load_checkpoint, wrap_fp16_model)
from mmdet.datasets import (build_dataloader, build_dataset, replace_ImageToTensor)
from mmdet.models import build_detector
from mmdet.utils.general_utils import write_images
from tools.lane_tools.tusimple.hough_lane import LaneEval


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--work-dir',
        help='the directory to save the file containing evaluation metrics')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
             'the inference speed')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
             'useful when you want to format the result to a specific format and '
             'submit it to the test server')
    parser.add_argument(
        '--eval',
        action='store_true',
        help='evaluation metrics, which depends on the dataset, e.g.'
             ' "Acc" for TuSimple, "F1" for CULane')
    parser.add_argument(
        '--data-dir', help='the directory to load testing images')
    parser.add_argument(
        '--show-dir', help='directory where painted images will be saved')
    parser.add_argument(
        '--show-score-thr',
        type=float,
        default=0.3,
        help='score threshold (default: 0.3)')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
             'workers, available when gpu-collect is not specified')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
             'in xxx=yyy format will be merged into config file. If the value to '
             'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
             'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
             'Note that the quotation marks are necessary and that no white space '
             'is allowed.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
             'format will be kwargs for dataset.evaluate() function (deprecate), '
             'change to --eval-options instead.')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
             'format will be kwargs for dataset.evaluate() function')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.eval_options:
        raise ValueError(
            '--options and --eval-options cannot be both '
            'specified, --options is deprecated in favor of --eval-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --eval-options')
        args.eval_options = args.options
    return args


def generate_culane_lines(out):
    out = out * (1640 / 640)  # B, 4, 360
    THRESHOLD = 30

    # out shape: 4, 360
    # (590, 1640, 3)
    lanes = []  # B, k, ny, 2
    for i in range(out.shape[0]):
        x = out[i, :]
        mask = x != 1640

        # Discard if no line
        if torch.count_nonzero(mask) >= THRESHOLD:
            y = torch.arange(0, len(x)).to(out.device) * (590 / 360)
            lane = torch.stack([x, y], dim=1)
            lane = lane[mask.nonzero(as_tuple=True)]
            lanes.append(lane)
    return lanes


def load_culane_img_data(path):
    with open(path, 'r') as data_file:
        img_data = data_file.readlines()
    img_data = [line.split() for line in img_data]
    img_data = [list(map(float, lane)) for lane in img_data]
    img_data = [[(lane[i], lane[i + 1]) for i in range(0, len(lane), 2)] for lane in img_data]
    img_data = [lane for lane in img_data if len(lane) >= 2]

    return img_data


def interp(points, n=50, sample=False):
    x = [x for x, _ in points]
    y = [y for _, y in points]

    # if sample:
    #     idx_list = np.linspace(0, len(x)-1, len(x)//10+1, dtype=np.int32)
    #     x = [x[i] for i in idx_list]
    #     y = [y[i] for i in idx_list]

    tck, u = splprep([x, y], s=0, t=n, k=min(3, len(points) - 1))

    u = np.linspace(0., 1., num=(len(u) - 1) * n + 1)
    return np.array(splev(u, tck)).T


def load_culane_data(data_dir, file_list_path):
    with open(file_list_path, 'r') as file_list:
        filepaths = [
            os.path.join(data_dir, line[1 if line[0] == '/' else 0:].rstrip().replace('.jpg', '.lines.txt'))
            for line in file_list.readlines()
        ]

    data = []
    for path in tqdm(filepaths):
        img_data = load_culane_img_data(path)
        data.append(img_data)

    return data


def draw_lane(lane, img=None, color=None, img_shape=None, width=30):
    if img is None:
        img = np.zeros(img_shape, dtype=np.uint8)
    lane = lane.astype(np.int32)
    for p1, p2 in zip(lane[:-1], lane[1:]):
        cv2.line(img, tuple(p1), tuple(p2), color=color, thickness=width)
    return img


def single_gpu_test(model, data_loader, data_dir, show_dir, training_test=False):
    model.eval()
    os.makedirs(pjoin(show_dir, 'culane'), exist_ok=True)
    colors = [[15, 121, 216], [115, 173, 16], [253, 160, 18], [252, 69, 69], [23, 183, 214], [223, 44, 113]]
       
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        torch.cuda.empty_cache()
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, training_test=training_test, **data)

            lane_map, idx_lanes = result[0], result[1]
            pre_lanes = torch.argmax(lane_map, dim=3)  # B, 5, 360, 640
            pre_index = torch.argmax(idx_lanes, dim=3).to(lane_map.device)  # B, 5, 360, 2
            pre_lanes[pre_index == 0] = 640  # B, 5, 360            
            pre_lanes = generate_culane_lines(pre_lanes[0])
            pre_lanes = [lane.data.cpu().numpy().tolist() for lane in pre_lanes]
            
            # prepare gt
            img_metas = data['img_metas'].data          
            img_file_path = img_metas[0][0]['filename']
            raw_file = img_metas[0][0]['sub_img_name']
            gt_lanes_path = img_file_path.replace('.jpg', '.lines.txt')
            gt_lanes = load_culane_img_data(gt_lanes_path)
                
            # save image
            img = cv2.imread(img_file_path)
            interp_pred = np.array([interp(pred_lane, n=5) for pred_lane in pre_lanes], dtype=object)  # (4, 50, 2)
            interp_anno = np.array([interp(anno_lane, n=5) for anno_lane in gt_lanes], dtype=object)  # (4, 50, 2)

            for k, lane_anno in enumerate(interp_anno):
                img = draw_lane(lane_anno, img=img, color=colors[4], width=10)
            for k, lane_pred in enumerate(interp_pred):
                img = draw_lane(lane_pred, img=img, color=colors[-1], width=5)

            show_img_path = pjoin(show_dir, raw_file)
            cv2.imwrite(show_img_path, img)       
        
        batch_size = len(result[0]) if isinstance(result, list) else len(result)
        for _ in range(batch_size):
            prog_bar.update()   
    print(' --->Done.')
    return


def main():
    args = parse_args()

    assert args.data_dir, ('Please specify (predict the images) with the argument "--data-dir"')
    assert args.show_dir, ('Please specify (save the results) with the argument "--show-dir"')

    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
        
    # update test data dir    
    cfg.batch_size = 1
    cfg.data['samples_per_gpu'] = cfg.batch_size
    cfg.data['workers_per_gpu'] = cfg.batch_size
    cfg.data_root = args.data_dir
    cfg.data.test['data_root'] = args.data_dir
    cfg.data.test['data_list'] = [args.data_dir + 'culane_test.txt']
    cfg.data.test['samples_per_gpu'] = 1

    cfg.model.pretrained = None
    if cfg.model.get('neck'):
        if isinstance(cfg.model.neck, list):
            for neck_cfg in cfg.model.neck:
                if neck_cfg.get('rfp_backbone'):
                    if neck_cfg.rfp_backbone.get('pretrained'):
                        neck_cfg.rfp_backbone.pretrained = None
        elif cfg.model.neck.get('rfp_backbone'):
            if cfg.model.neck.rfp_backbone.get('pretrained'):
                cfg.model.neck.rfp_backbone.pretrained = None

    # in case the test dataset is concatenated
    samples_per_gpu = 1
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
        if samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        samples_per_gpu = max(
            [ds_cfg.pop('samples_per_gpu', 1) for ds_cfg in cfg.data.test])
        if samples_per_gpu > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    rank, _ = get_dist_info()
    # allows not to create
    if args.work_dir is not None and rank == 0:
        mmcv.mkdir_or_exist(osp.abspath(args.work_dir))

    if args.show_dir is not None and rank == 0:
        mmcv.mkdir_or_exist(osp.abspath(args.show_dir))

    # build the dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False)

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)
    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES

    model = MMDataParallel(model, device_ids=[0])
    single_gpu_test(model, data_loader, 
                    data_dir=args.data_dir, show_dir=args.show_dir, training_test=True)
    # single_gpu_test(model, data_loader, cfg.data.test.data_root, cfg.work_dir, training_test=False)
    


if __name__ == '__main__':
    main()
