import json
import random

import cv2
import numpy as np
from scipy.interpolate import splprep, splev
from mmdet.utils.general_utils import mkdir, path_join

import mmcv
import torch
from torch import nn
import torch.nn.functional as F

from collections import OrderedDict
from .custom import CustomDataset
from .pipelines import Compose
from .builder import DATASETS

from tools.lane_tools.tusimple.hough_lane import LaneEval
from mmdet.utils.general_utils import write_images


@DATASETS.register_module
class TuSimpleDataset(CustomDataset):
    def __init__(self, data_root, data_list, pipeline, max_num_lane, ori_scale, img_scale, h_samples, cp_work_dir=None,
                 samples_per_gpu=None, test_mode=False, test_suffix='.jpg'):
   
        self.img_prefix = data_root
        self.data_list = data_list
        self.test_suffix = test_suffix
        self.test_mode = test_mode
        self.max_num_lane = max_num_lane
        self.ori_scale = ori_scale
        self.img_scale = img_scale
        self.h_samples = h_samples
        self.cp_work_dir = cp_work_dir

        if self.cp_work_dir is not None:
            mkdir(self.cp_work_dir)

        # for checkpoint visualization
        self.cp_img_postfix = 1

        # read image list
        self.img_infos = self.parser_datalist(data_list)
        # set group flag for the sampler
        if not self.test_mode:
            self._set_group_flag()
        self.pipeline = Compose(pipeline)

    def parser_datalist(self, data_list):
        img_infos = []
        for anno_file in data_list:
            json_gt = [json.loads(line) for line in open(anno_file)]
            img_infos += json_gt
        return img_infos

    def extend_lane(self, old_points, th=50):
        # end point in the image center
        x_last, y_last = old_points[-1, 0], old_points[-1, 1]
        if th < x_last < self.ori_scale[0] - th and y_last < self.ori_scale[1] - th:
            return old_points

        points = []
        if old_points[-1, 1] < self.ori_scale[1] - 1:
            gap = (old_points[-1, 0] - old_points[-3, 0]) / (old_points[-1, 1] - old_points[-3, 1])
            for k in range(old_points[-1, 1] + 1, self.ori_scale[1]):
                x_coor = int(old_points[-1, 0] + gap * (k - old_points[-1, 1]))
                if x_coor >= self.ori_scale[0] or x_coor < 0:
                    break
                points.append([x_coor, k])
        points = np.concatenate((old_points, np.array(points))) if len(points) > 0 else old_points
        return points

    def load_labels(self, idx):
        lanes = self.img_infos[idx]['lanes']
        h_samples = self.img_infos[idx]['h_samples']

        keypoints, class_labels = [], []
        for i, lane in enumerate(lanes):
            points = np.array([[x, y] for (x, y) in zip(lane, h_samples) if x >= 0])
            if len(points) < 3:
                continue
            new_points = self.extend_lane(np.array(points))
            inter_y = np.arange(new_points[0, 1], new_points[-1, 1] + 1, 1)
            inter_x = np.interp(inter_y, new_points[:, 1], new_points[:, 0])
            tkps = [[x, y] for x, y in zip(inter_x, inter_y)]
            tclab = [str(i)] * len(tkps)

            keypoints.extend(tkps)
            class_labels.extend(tclab)

        return keypoints, class_labels

    def prepare_train_img(self, idx):
        sub_img_name = self.img_infos[idx]['raw_file']
        imgname = path_join(self.img_prefix, sub_img_name)
        img = cv2.imread(imgname)

        ori_shape = img.shape
        keypoints, class_labels = self.load_labels(idx)
        results = dict(
            filename=imgname,
            sub_img_name=sub_img_name,
            img=img,
            keypoints=keypoints,
            class_labels=class_labels,
            img_shape=ori_shape,
            ori_shape=ori_shape,
        )
        return self.pipeline(results)

    def prepare_test_img(self, idx):
        sub_img_name = self.img_infos[idx]['raw_file']
        imgname = path_join(self.img_prefix, sub_img_name)
        img = cv2.imread(imgname)

        ori_shape = img.shape
        keypoints, class_labels = self.load_labels(idx)
        results = dict(
            filename=imgname,
            sub_img_name=sub_img_name,
            img=img,
            keypoints=keypoints,
            class_labels=class_labels,
            img_shape=ori_shape,
            ori_shape=ori_shape,
        )
        return self.pipeline(results)

    def _set_group_flag(self):
        """Set flag according to image aspect ratio.

        Original:
        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0.

        Current:
        All images will be set 1
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)
        for i in range(len(self)):
            self.flag[i] = 1

    def __len__(self):
        return len(self.img_infos)

    def format_results(self, results, **kwargs):
        """Place holder to format result to dataset specific output."""

    def _get_predict_lanes(self, results, total_num):
        """
        Args:
            results: [lane_map, idx_lanes, seg_maps, aug_weight]
        """
        img_w, img_h = self.img_scale  # img_scale = (640, 360)
        idx_list = [int(idx / 720 * img_h) for idx in self.h_samples]

        output_path = path_join(self.cp_work_dir, "temp_eval.txt")
        temp_dict = {"h_samples": self.h_samples, "run_time": 30}

        print(f"Evaluation: total {total_num} samples.")
        prog_bar = mmcv.ProgressBar(total_num)
        with open(output_path, 'w') as fp:
            k = 0
            for data in results:
                lane_map, idx_lanes = data[0], data[1]

                # get lanes
                pre_lanes = torch.argmax(lane_map, dim=3)  # B, 5, 360, 640
                pre_index = torch.argmax(idx_lanes, dim=3).to(lane_map.device)  # B, 5, 360, 2
                pre_lanes[pre_index == 0] = img_w  # B, 5, 360

                # get points for cpu
                pre_lanes_cpu = pre_lanes.data.cpu().numpy()  # B, 5, 360
                pre_lanes_cpu = pre_lanes_cpu / img_w * 1280
                pre_lanes_cpu[pre_lanes_cpu == 1280] = -2
                pre_lanes_cpu = pre_lanes_cpu.tolist()  # B, 5, 360

                for lanes in pre_lanes_cpu:
                    # drop noise by image size
                    lanes = [lane for lane in lanes if sum(lane) != -2 * len(lane)]
                    lanes = [[lane[idx] for idx in idx_list] for lane in lanes]

                    # save to file
                    temp_dict['raw_file'] = self.img_infos[k]['raw_file']
                    temp_dict['lanes'] = lanes
                    json_str = json.dumps(temp_dict)
                    fp.write(json_str + '\n')
                    k = k + 1
                    prog_bar.update()

        # val & test only one json file
        assert len(self.data_list) == 1, "Only one json file in testing"
        res = LaneEval.bench_one_submit(output_path, self.data_list[0])
        # os.remove(output_path)
        return res

    def _training_stage_display(self, result):
        img_w, img_h = self.img_scale  # img_scale = (640, 360)
        lane_map, idx_lanes, seg_map, hough_map, points_list, \
        gt_img, gt_segment_map, gt_hough_map, gt_points_list, gt_line_map, line_map = result

        pre_lanes = torch.argmax(lane_map, dim=3)  # B, 5, 360, 640
        pre_index = torch.argmax(idx_lanes, dim=3).to(lane_map.device)  # B, 5, 360, 2
        pre_lanes[pre_index == 0] = img_w  # B, 5, 360

        # for BCEWithLogitsLoss
        seg_map = torch.sigmoid(seg_map)
        hough_map = torch.sigmoid(hough_map)
        lane_maps = torch.sigmoid(lane_map)
        line_map = torch.sigmoid(line_map)

        hough_map_with_points = hough_map.expand(-1, 3, -1, -1).permute(0, 2, 3, 1).cpu().numpy()
        for batch_idx in range(len(hough_map_with_points)):
            img = hough_map_with_points[batch_idx]
            points = points_list[batch_idx]
            for y, x in points:
                cv2.circle(img, center=(int(x), int(y)), radius=1,
                           color=(1, 0, 0), thickness=cv2.FILLED, lineType=cv2.LINE_AA)

            points = gt_points_list[batch_idx]
            for y, x in points:
                cv2.circle(img, center=(int(x), int(y)), radius=1,
                           color=(0, 0, 1), thickness=cv2.FILLED, lineType=cv2.LINE_AA)
        hough_map_with_points = torch.from_numpy(hough_map_with_points).to(hough_map.device).permute(0, 3, 1, 2)

        # prepare
        disp_lane = [gt_img, pre_lanes]
        disp_map = [gt_segment_map, lane_maps, seg_map, gt_line_map, line_map]
        disp_hough = [gt_hough_map, hough_map, hough_map_with_points]
        write_images([disp_lane, disp_map, disp_hough], self.cp_work_dir, f"{self.cp_img_postfix:05d}",
                     img_w=img_w, img_h=img_h)
        self.cp_img_postfix += 1
        return

    def _testing_stage_display(self, results, show_dir, total_num):
        return

    def evaluate(self, results, metric=None, logger=None, proposal_nums=None, iou_thr=None, scale_ranges=None,
                 training_test=True, show=False, show_dir=None):
        total_num = len(results)

        # reshape to batch_size = 1
        if len(results) > 0 and len(results[0][0]) > 1:
            total_num = [len(results[i][0]) for i in range(total_num)]
            total_num = sum(total_num)

        # display intermediate images
        if training_test:
            idx = random.randint(0, len(results) - 1)  # random select one to display
            self._training_stage_display(results[idx])
        else:
            # visualization test results
            if show or show_dir:
                # TODO unimplemented
                # self._testing_stage_display(results, show_dir, total_num)
                pass

        # eval metric
        eval_results = OrderedDict()
        res = self._get_predict_lanes(results, total_num)
        for r in json.loads(res):
            eval_results[r["name"]] = r["value"]
        return eval_results
