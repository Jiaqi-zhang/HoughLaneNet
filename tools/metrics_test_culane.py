import argparse
import os
import os.path as osp
import pickle
import time
import warnings

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
from tools.lane_tools.culane.culane_metric import eval_predictions


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


def generate_culane_lines(out):
    out = out * (1640 / 640)  # B, 4, 360
    THRESHOLD = 60
    # THRESHOLD = 10

    # out shape: 4, 360
    # (590, 1640, 3)
    lanes = []  # B, k, ny, 2
    for i in range(out.shape[0]):
        x = out[i, :]
        mask = x != 1640

        # Discard if no line
        if torch.count_nonzero(mask) >= THRESHOLD:
            y = torch.arange(0, len(x)) * (590 / 360)
            lane = torch.stack([x, y], dim=1)
            lane = lane[mask.nonzero(as_tuple=True)]
            lanes.append(lane)
    return lanes


def write_culane_data(test_data, save_predict_path):
    names, pre_lanes = test_data

    for i, name in enumerate(names):
        lanes = generate_culane_lines(pre_lanes[i])
        spred_file_path = os.path.join(save_predict_path, os.path.split(name)[0])
        os.makedirs(spred_file_path, exist_ok=True)

        with open(os.path.join(spred_file_path, f"{os.path.split(name)[-1][0:-4]}.lines.txt"), 'w') as fp:
            for lane in lanes:
                lane_str = " ".join([f"{x:.4f} {y:.4f}" for [x, y] in lane])
                fp.write(lane_str + '\n')


def single_gpu_test(model, data_loader, anno_dir, out_dir=None, training_test=False):
    model.eval()
    results = []
    total_num, total_run_time = 0, 0

    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        torch.cuda.empty_cache()
        with torch.no_grad():
            tic = time.time()
            result = model(return_loss=False, rescale=True, training_test=training_test, **data)
            toc = time.time()

            lane_map, idx_lanes = result[0], result[1]
            pre_lanes = torch.argmax(lane_map, dim=3)  # B, 5, 360, 640
            pre_index = torch.argmax(idx_lanes, dim=3).to(lane_map.device)  # B, 5, 360, 2
            pre_lanes[pre_index == 0] = 640  # B, 5, 360

            # get lanes
            img_metas = data['img_metas'].data
            names = [img_meta['sub_img_name'] for img_meta in img_metas[0]]
            total_run_time += (toc - tic) * 1000
            total_num += len(names)

        results.append([names, pre_lanes.cpu()])
        batch_size = len(result[0]) if isinstance(result, list) else len(result)

        for _ in range(batch_size):
            prog_bar.update()

    # get metrics
    save_predict_path = os.path.join(out_dir, PRED_SAVE_DIR)
    os.makedirs(save_predict_path, exist_ok=True)

    with mp.Pool(processes=4) as pool:
        f = partial(write_culane_data, save_predict_path=save_predict_path)
        for _ in pool.imap_unordered(f, results): pass

    eval_result = eval_predictions(pred_dir=save_predict_path,
                                   anno_dir=anno_dir,
                                   list_path=os.path.join(anno_dir, ANNO_TXT_PATH),
                                   official=True,
                                   sequential=False)
    eval_result['FPS'] = 1000. / (total_run_time / total_num)

    header = '=' * 40
    print(header)
    for metric, value in eval_result.items():
        if isinstance(value, float):
            print('{}: {:.4f}'.format(metric, value))
        else:
            print('{}: {}'.format(metric, value))
    print(header)
    return


def multi_gpu_test(model, data_loader, anno_dir, out_dir, training_test=False):
    model.eval()
    results = []
    total_num, total_run_time = 0, 0

    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    time.sleep(2)  # This line can prevent deadlock problem in some cases.
    for i, data in enumerate(data_loader):
        torch.cuda.empty_cache()
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
            total_run_time += (toc - tic) * 1000
            total_num += len(names)
        results.append([names, pre_lanes.cpu()])

        if rank == 0:
            batch_size = len(result[0]) if isinstance(result, list) else len(result)
            for _ in range(batch_size * world_size):
                prog_bar.update()

    save_predict_path = os.path.join(out_dir, PRED_SAVE_DIR)
    os.makedirs(save_predict_path, exist_ok=True)

    with mp.Pool(processes=4) as pool:
        f = partial(write_culane_data, save_predict_path=save_predict_path)
        jobs = pool.imap_unordered(f, results)
        if rank == 0:
            jobs = tqdm.tqdm(jobs, total=len(results))
        for _ in jobs: pass

    if rank == 0:
        eval_result = eval_predictions(pred_dir=save_predict_path,
                                       anno_dir=anno_dir,
                                       list_path=os.path.join(anno_dir, ANNO_TXT_PATH),
                                       official=True,
                                       sequential=False)
        eval_result['FPS'] = 1000. / (total_run_time / total_num)

        header = '=' * 40
        print(header)
        for metric, value in eval_result.items():
            if isinstance(value, float):
                print('{}: {:.4f}'.format(metric, value))
            else:
                print('{}: {}'.format(metric, value))
        print(header)
    return


def main():
    args = parse_args()

    # assert args.out or args.eval or args.format_only or args.show \
    #        or args.show_dir, \
    #     ('Please specify at least one operation (save/eval/format/show the '
    #      'results / save the results) with the argument "--out", "--eval"'
    #      ', "--format-only", "--show" or "--show-dir"')

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
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

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

    if not distributed:
        model = MMDataParallel(model, device_ids=[0])
        single_gpu_test(model, data_loader, cfg.data.test.data_root, cfg.work_dir, training_test=False)
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        multi_gpu_test(model, data_loader, cfg.data.test.data_root, cfg.work_dir, training_test=False)


if __name__ == '__main__':
    ANNO_TXT_PATH = './list/test.txt'
    # ANNO_TXT_PATH = './list/test_split/test7_cross.txt'
    PRED_SAVE_DIR = 'lane_text'
    main()
