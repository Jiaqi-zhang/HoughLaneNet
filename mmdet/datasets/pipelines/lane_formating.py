import math
import numpy as np
from scipy.optimize import leastsq
import warnings

warnings.filterwarnings('ignore')
warnings.simplefilter('ignore', np.RankWarning)

from mmcv.parallel import DataContainer as DC

from ..builder import PIPELINES
from .hough_util import line_to_hough_space, hough_points_to_line
from .formating import Collect, to_tensor


def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def gaussian1D(width=3):
    sigma = (2 * width + 1) / 6
    x = np.arange(-1 * width, width + 1)
    v = np.exp(-(x * x) / (2 * sigma * sigma))
    return v.tolist()


def draw_umich_gaussian(map, center, radius, gaussian):
    # if gaussian is None:
    #     diameter = 2 * radius + 1
    #     gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)
    y, x = int(center[0]), int(center[1])  # angle, rho
    height, width = map.shape[0:2]
    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    masked_map = map[y - top:y + bottom, x - left:x + right]
    if min(masked_gaussian.shape) > 0 and min(masked_map.shape) > 0:
        np.maximum(masked_map, masked_gaussian, out=masked_map)
    return map


def fit_line_by_leastsq(X, Y):
    # dist = y * sin(theta) + x * cos(theta)
    def error(p, x, y):
        return y * math.sin(p[0]) + x * math.cos(p[0]) - p[1]

    # para = leastsq(error, x0=[0, 1], args=(X, Y))
    para = leastsq(error, x0=[np.pi / 2, 1], args=(X, Y))
    theta, dist = para[0][0], para[0][1]
    # print("theta=", theta, '\n', "dist=", dist)
    return theta, dist


@PIPELINES.register_module
class CollectLane(Collect):

    def __init__(
            self,
            line_width,
            line_mode,
            max_num_lane,
            num_angle,
            num_rho,
            hough_point_radius,
            hough_point_ratio,
            keys,
            meta_keys=('filename', 'ori_shape', 'img_shape', 'img_norm_cfg'),
    ):
        super(CollectLane, self).__init__(keys, meta_keys)
        self.line_width = line_width
        self.line_mode = line_mode
        self.max_num_lane = max_num_lane
        self.num_angle = num_angle
        self.num_rho = num_rho
        self.hough_point_radius = hough_point_radius
        self.hough_point_ratio = hough_point_ratio
        self.normal_v = gaussian1D(width=self.line_width)

        diameter = 2 * self.hough_point_radius + 1
        self.normal_hough = gaussian2D((diameter, diameter), sigma=diameter / 6)

        diameter = 2 * self.line_width + 1
        self.normal_line = gaussian2D((diameter, diameter), sigma=diameter / 6)

    def cons_maps_by_line(self, segment_map, hough_map, line_map, clc_lane, point_list, img_w, img_h):
        # get hough map
        hough_point_list = []
        for i in range(self.max_num_lane):
            idx = clc_lane[i, :] != img_w
            if np.sum(idx) >= 2:
                x = clc_lane[i, idx]
                y = np.argwhere(clc_lane[i, :] != img_w).reshape(-1, )

                line = [y[0], x[0], y[-1], x[-1]]
                angle, rho = line_to_hough_space(line, self.num_angle, self.num_rho, (img_h, img_w))
                assert 0 <= angle < self.num_angle
                assert 0 <= rho < self.num_rho
                hough_point_list.append({'angle': angle, 'rho': rho, 'clc_idx': i})
                hough_map = draw_umich_gaussian(hough_map, (angle, rho), self.hough_point_radius, gaussian=self.normal_hough)

                # fill line map
                inter_y = np.linspace(0, img_h-1, img_h * 10)
                if x[0] != x[-1]:
                    eq = np.poly1d(np.polyfit([y[0], y[-1]], [x[0], x[-1]], 1))
                    inter_x = eq(inter_y)
                else:
                    inter_x = [x[0]] * len(inter_y)

                for _x, _y in zip(inter_x, inter_y):
                    _x, _y = int(_x), int(_y)
                    if _x < 0 or _x >= img_w:
                        continue
                    line_map = draw_umich_gaussian(line_map, (_y, _x), self.line_width, gaussian=self.normal_line)
            else:
                segment_map[i] = 0

        hough_point_list.sort(key=lambda p: p['angle'] * 1e4 + p['rho'])
        clc_idx_list = [p['clc_idx'] for p in hough_point_list]

        # Resort according to clc_lane_idx
        if len(clc_idx_list) > 0:
            segment_map_resort = np.stack([segment_map[i] for i in clc_idx_list], axis=0)
            segment_map[:len(segment_map_resort)] = segment_map_resort
            segment_map[len(segment_map_resort):] = 0

            for i, p in enumerate(hough_point_list):
                point_list[i, 0], point_list[i, 1] = p['angle'], p['rho']
        return segment_map, line_map, hough_map, point_list

    def cons_maps_by_hough(self, segment_map, hough_map, line_map, clc_lane, point_list, img_w, img_h):
        hough_point_list = []
        for i in range(self.max_num_lane):
            idx = clc_lane[i, :] != img_w
            if np.sum(idx) >= 2:
                x = clc_lane[i, idx]
                y = np.argwhere(clc_lane[i, :] != img_w).reshape(-1, )

                angle_list, rho_list = [], []
                idx_list = list(range(0, len(x), 4))
                if len(idx_list) < 2:
                    idx_list.append(len(x) - 1)

                # only select 20 points
                idx_list.reverse()
                for c in range(0, 10):
                    if c >= len(idx_list) - 1:
                        break
                    s_c = idx_list[c]
                    e_c = idx_list[c + 1]
                    line = [y[s_c], x[s_c], y[e_c], x[e_c]]
                    angle, rho = line_to_hough_space(line, self.num_angle, self.num_rho, (img_h, img_w))
                    assert 0 <= angle < self.num_angle
                    assert 0 <= rho < self.num_rho
                    angle_list.append(angle)
                    rho_list.append(rho)

                angle, rho = np.mean(angle_list), np.mean(rho_list)
                hough_point_list.append({'angle': angle, 'rho': rho, 'clc_idx': i})

                # def crop_rho(r, r_min=self.num_rho*1//4, r_max=self.num_rho*3//4):
                #     t = (r - r_min) / (r_max - r_min)
                #     r_cropped = np.clip(t * self.num_rho, 0, self.num_rho - 1)
                #     return r_cropped
                hough_map = draw_umich_gaussian(hough_map, (angle, rho), self.hough_point_radius, gaussian=self.normal_hough)
            else:
                segment_map[i] = 0

        hough_point_list.sort(key=lambda p: p['angle'] * 1e4 + p['rho'])
        clc_idx_list = [p['clc_idx'] for p in hough_point_list]

        # Resort according to clc_lane_idx
        if len(clc_idx_list) > 0:
            segment_map_resort = np.stack([segment_map[i] for i in clc_idx_list], axis=0)
            segment_map[:len(segment_map_resort)] = segment_map_resort
            segment_map[len(segment_map_resort):] = 0

            for i, p in enumerate(hough_point_list):
                point_list[i, 0], point_list[i, 1] = p['angle'], p['rho']

        # for line map
        line_points = hough_points_to_line([[p['angle'], p['rho']] for p in hough_point_list],
                                           self.num_angle, self.num_rho, (img_h, img_w))
        for point in line_points:
            y0, x0, y1, x1 = point

            # fill line map
            inter_y = np.linspace(0, img_h-1, img_h * 5)
            if x0 != x1:
                eq = np.poly1d(np.polyfit([y0, y1], [x0, x1], 1))
                inter_x = eq(inter_y)
            else:
                inter_x = [x0] * len(inter_y)

            for _x, _y in zip(inter_x, inter_y):
                _x, _y = int(_x), int(_y)
                if _x < 0 or _x >= img_w:
                    continue
                line_map = draw_umich_gaussian(line_map, (_y, _x), self.line_width, gaussian=self.normal_line)
        return segment_map, line_map, hough_map, point_list

    def target(self, results):
        keypoints = results['keypoints']
        class_labels = results['class_labels']

        # init inputs
        img_h, img_w, _ = results['img_shape']  # (360, 640, 3)
        # clc_lane: [L, H], value: [0, W)
        clc_lane = np.ones((self.max_num_lane, img_h), dtype=np.float32) * img_w
        # segment_map: [L, H, W], value: {0.0, 1.0}
        segment_map = np.zeros((self.max_num_lane, img_h, img_w), dtype=np.float32)
        line_map = np.zeros((img_h, img_w), dtype=np.float32)

        # hough_map: [NumAngle, NumRho], value: [0.0, 1.0]
        hough_map = np.zeros((self.num_angle, self.num_rho), dtype=np.float32)
        point_list = np.ones((self.max_num_lane, 2), dtype=np.float32) * -1

        # get lane points
        for i, point in enumerate(keypoints):
            clc_idx = int(class_labels[i])
            x, y = int(point[0]), int(point[1])
            segment_map[clc_idx] = draw_umich_gaussian(segment_map[clc_idx], (y, x), self.line_width, gaussian=self.normal_line)
            clc_lane[clc_idx, y] = x
        segment_map = np.clip(segment_map, a_min=0.0, a_max=1.0)

        if self.line_mode == 'hough':
            segment_map, line_map, hough_map, point_list = self.cons_maps_by_hough(segment_map, hough_map, line_map,
                                                                                   clc_lane, point_list, img_w, img_h)
        else:
            segment_map, line_map, hough_map, point_list = self.cons_maps_by_line(segment_map, hough_map, line_map,
                                                                                  clc_lane, point_list, img_w, img_h)

        results['segment_map'] = segment_map
        results['line_map'] = np.expand_dims(line_map, 0)
        results['hough_map'] = np.expand_dims(hough_map, 0)  # [1, NumAngle, NumRho]
        results['point_list'] = point_list
        return True

    def __call__(self, results):
        """Call function to collect keys in results. The keys in ``meta_keys``
        will be converted to :obj:mmcv.DataContainer.

        Args:
            results (dict): Result dict contains the data to collect.

        Returns:
            dict: The result dict contains the following keys

                - keys in``self.keys``
                - ``img_metas``
        """

        data = {}
        img_meta = {}

        # last process for prepare data
        valid = self.target(results)
        if not valid:
            return None

        for key in self.meta_keys:
            img_meta[key] = results[key]
        data['img_metas'] = DC(img_meta, cpu_only=True)
        for key in self.keys:
            data[key] = results[key]
        return data
