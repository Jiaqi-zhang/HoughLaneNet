import argparse
import os
import os.path as osp
import pickle
import time
import warnings
import ujson as json

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)

# from mmdet.apis import multi_gpu_test, single_gpu_test
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.models import build_detector

import tqdm
from functools import partial
import torch.distributed as dist
import torch.multiprocessing as mp
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
    parser.add_argument('--show', action='store_true', help='show results')
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
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
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


def write_tusimple_data(test_data, show, show_dir, save_predict_path):
    h_samples = [160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300, 310, 320, 330, 340, 350,
                 360, 370, 380, 390, 400, 410, 420, 430, 440, 450, 460, 470, 480, 490, 500, 510, 520, 530, 540, 550,
                 560, 570, 580, 590, 600, 610, 620, 630, 640, 650, 660, 670, 680, 690, 700, 710]
    temp_dict = {"h_samples": h_samples}
    list_names, list_run_time, list_pre_lanes = test_data

    with open(save_predict_path, 'a') as fp:
        for name, run_time, pre_lanes in zip(list_names, list_run_time, list_pre_lanes):
            # get points for cpu
            pre_lanes = pre_lanes / 640 * 1280  # B, 5, 360
            pre_lanes[pre_lanes == 1280] = -2
            pre_lanes = pre_lanes.tolist()  # B, 5, 360

            # drop noise
            lanes = [lane[80:] for lane in pre_lanes if sum(lane) != -2 * len(lane)]
            lanes = [lane[0:280:5] for lane in lanes]

            # save to file
            temp_dict['raw_file'] = name
            temp_dict['lanes'] = lanes
            temp_dict['run_time'] = run_time
            json_str = json.dumps(temp_dict)
            fp.write(json_str + '\n')
    return


def single_gpu_test(model, data_loader, gt_data_list, out_dir=None, show=False, show_dir=None, training_test=False):
    model.eval()
    results = []

    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    torch.cuda.empty_cache()
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            tic = time.perf_counter()
            result = model(return_loss=False, rescale=True, training_test=training_test, **data)
            toc = time.perf_counter()

            lane_map, idx_lanes = result[0], result[1]
            pre_lanes = torch.argmax(lane_map, dim=3)  # B, 5, 360, 640
            pre_index = torch.argmax(idx_lanes, dim=3).to(lane_map.device)  # B, 5, 360, 2
            pre_lanes[pre_index == 0] = 640  # B, 5, 360

            # get lanes
            img_metas = data['img_metas'].data
            names = [img_meta['sub_img_name'] for img_meta in img_metas[0]]
            run_times = [(toc - tic) * 1000 / len(names)] * len(names)

        results.append([names, run_times, pre_lanes.data.cpu().numpy()])
        batch_size = len(result[0]) if isinstance(result, list) else len(result)

        for _ in range(batch_size):
            prog_bar.update()

    with mp.Pool(processes=4) as pool:
        f = partial(write_tusimple_data, show=show, show_dir=show_dir, save_predict_path=out_dir)
        for _ in pool.imap_unordered(f, results): pass

    eval_result = LaneEval.bench_one_submit(out_dir, gt_data_list[0])
    print(f"\n==============Eval==============")
    for r in json.loads(eval_result):
        metric, value = r["name"], r["value"]
        if isinstance(value, float):
            print('{}: {:.4f}'.format(metric, value))
        else:
            print('{}: {}'.format(metric, value))
    print(f"==============Eval==============\n")
    return


def main():
    args = parse_args()

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

    # init distributed env first, since logger depends on the dist info.
    distributed = False

    rank, _ = get_dist_info()
    # allows not to create
    if args.work_dir is not None and rank == 0:
        mmcv.mkdir_or_exist(osp.abspath(args.work_dir))

    # build the dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
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

    save_predict_path = os.path.join(cfg.work_dir, "temp_eval.txt")
    if rank == 0 and os.path.exists(save_predict_path):
        os.remove(save_predict_path)

    model = MMDataParallel(model, device_ids=[0])
    single_gpu_test(model, data_loader, cfg.data.test.data_list, save_predict_path,
                    show=args.show, show_dir=args.show_dir, training_test=False)



if __name__ == '__main__':
    main()
