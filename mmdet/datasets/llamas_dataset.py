import json
import math
import os
import glob
import random
import pickle as pkl

import cv2
import numpy as np
from mmdet.utils.general_utils import mkdir, getPathList, path_join

import mmcv
import torch
from torch import nn
import torch.nn.functional as F

from collections import OrderedDict
from .custom import CustomDataset
from .pipelines import Compose
from .builder import DATASETS

from mmdet.utils.general_utils import write_images


@DATASETS.register_module
class LLAMASDataset(CustomDataset):
    def __init__(self, data_root, data_json_dir, data_list, pipeline, max_num_lane, ori_scale, img_scale, mode,
                 cp_work_dir=None, samples_per_gpu=None, test_mode=False, test_suffix='.jpg'):
        self.img_prefix = data_root
        self.mode = mode
        self.data_json_dir = data_json_dir
        self.data_list = data_list
        self.max_num_lane = max_num_lane
        self.test_mode = test_mode
        self.ori_scale = ori_scale
        self.img_scale = img_scale
        self.cp_work_dir = cp_work_dir
        self.test_suffix = test_suffix

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
        # Waiting for the dataset to load is tedious, let's cache it
        os.makedirs('cache', exist_ok=True)
        cache_path = f'cache/llamas_{self.data_json_dir}_{self.mode}.pkl'
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as cache_file:
                return pkl.load(cache_file)

        img_infos = []
        for anno_file in data_list:
            line_gt = [line.strip() for line in open(anno_file)]

            js_data = []
            for file_path in line_gt:
                dirs = file_path.split('/')
                js_file_path = f"{dirs[2]}_{dirs[3].replace('.png', '.json').replace('_color_rect', '')}"
                ab_js_path = os.path.join(self.img_prefix, self.data_json_dir, js_file_path)
                temp_data = [json.loads(line) for line in open(ab_js_path)][0]
                js_data.append(temp_data)
            img_infos.extend(js_data)

        with open(cache_path, 'wb') as cache_file:
            pkl.dump(img_infos, cache_file)
        return img_infos

    def extend_lane(self, old_points, th=50):
        # end point in the image center
        x_last, y_last = old_points[-1, 0], old_points[-1, 1]
        if th < x_last < self.ori_scale[0] - th and y_last < self.ori_scale[1] - th:
            return old_points

        points = []
        if old_points[-1, 1] < self.ori_scale[1] - 1:
            gap = (old_points[-1, 0] - old_points[-3, 0]) / (old_points[-1, 1] - old_points[-3, 1])
            for k in range(int(old_points[-1, 1] + 1), self.ori_scale[1]):
                x_coor = int(old_points[-1, 0] + gap * (k - old_points[-1, 1]))
                if x_coor >= self.ori_scale[0] or x_coor < 0:
                    break
                points.append([x_coor, k])
        points = np.concatenate((old_points, np.array(points))) if len(points) > 0 else old_points
        return points

    def load_train_labels(self, idx):
        lanes = self.img_infos[idx]['lanes']
        h_samples = self.img_infos[idx]['h_samples']

        keypoints, class_labels = [], []
        for i, lane in enumerate(lanes):
            points = np.array([[x, y] for (x, y) in zip(lane, h_samples) if 0 <= x < self.ori_scale[0]])
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
        keypoints, class_labels = self.load_train_labels(idx)
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

    def load_test_labels(self, idx):
        lanes = self.img_infos[idx]['lanes']
        h_samples = self.img_infos[idx]['h_samples']

        keypoints, class_labels = [], []
        for i, lane in enumerate(lanes):
            points = np.array([[x, y] for (x, y) in zip(lane, h_samples) if 0 <= x < self.ori_scale[0]])
            if len(points) < 3:
                continue
            # new_points = self.extend_lane(np.array(points))
            new_points = np.array(points)
            inter_y = np.arange(new_points[0, 1], new_points[-1, 1] + 1, 1)
            inter_x = np.interp(inter_y, new_points[:, 1], new_points[:, 0])
            tkps = [[x, y] for x, y in zip(inter_x, inter_y)]
            tclab = [str(i)] * len(tkps)

            keypoints.extend(tkps)
            class_labels.extend(tclab)
        return keypoints, class_labels

    def prepare_test_img(self, idx):
        sub_img_name = self.img_infos[idx]['raw_file']
        imgname = path_join(self.img_prefix, sub_img_name)
        img = cv2.imread(imgname)

        ori_shape = img.shape
        keypoints, class_labels = self.load_test_labels(idx)
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

    def _training_stage_display(self, result):
        img_w, img_h = self.img_scale  # img_scale = (640, 360)
        lane_map, idx_lanes, seg_map, hough_map, points_list, \
        gt_img, gt_segment_map, gt_hough_map, gt_points_list = result

        pre_lanes = torch.argmax(lane_map, dim=3)  # B, 5, 360, 640
        pre_index = torch.argmax(idx_lanes, dim=3).to(lane_map.device)  # B, 5, 360, 2
        pre_lanes[pre_index == 0] = img_w  # B, 5, 360

        # for BCEWithLogitsLoss
        seg_map = torch.sigmoid(seg_map)
        # hough_line_map = torch.sigmoid(hough_line_map)
        hough_map = torch.sigmoid(hough_map)
        lane_maps = torch.sigmoid(lane_map)

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
        disp_map = [gt_segment_map, lane_maps, seg_map]
        disp_hough = [gt_hough_map, hough_map, hough_map_with_points]
        write_images([disp_lane, disp_map, disp_hough], self.cp_work_dir, f"{self.cp_img_postfix:05d}")
        self.cp_img_postfix += 1
        return

    def evaluate(self, results, metric=None, logger=None, proposal_nums=None, iou_thr=None, scale_ranges=None,
                 training_test=True, show=False, show_dir=None):
        # Fix: After resume, intermediate results are not saved
        # Evaluation is only called during the training phase
        idx = random.randint(0, len(results) - 1)  # random select one to display
        self._training_stage_display(results[idx])

        # Note: training phase does not test metric
        # eval metric
        eval_results = OrderedDict({'TP': 0, 'FP': 0, 'FN': 0, 'Precision': 0.0, 'Recall': 0.0, 'F1': 0.0})
        return eval_results
