import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from mmcv.cnn import ConvModule, ConvTranspose2d
from mmcv.cnn import Linear
from mmcv.utils import to_2tuple
from mmcv.runner.base_module import BaseModule, Sequential
from einops import rearrange
from skimage.measure import label, regionprops

from ..builder import HEADS
from ..utils.dht import DHTLayer, RHTLayer


class InstanceHead(BaseModule):

    def __init__(self, in_dim, out_dim, groups=1, init_cfg=None):
        super(InstanceHead, self).__init__(init_cfg)
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.groups = groups
        assert in_dim % self.groups == 0, "instance in_dim must be divisible by groups"

    @property
    def dim_param(self):
        return self.out_dim * (self.in_dim // self.groups)

    def forward(self, feat, params):
        # feat: [B, D, H, W]
        # params: [B, max_num_lane, P]
        B, N, P = params.shape
        params = rearrange(params,
                           'b n (cOut cIn kH kW) -> b n cOut cIn kH kW',
                           cOut=self.out_dim,
                           kH=1,
                           kW=1)

        batch_tensors = []
        for batch_idx in range(B):
            batch_tensors.append(
                torch.cat([
                    F.conv2d(
                        feat[batch_idx:batch_idx + 1],
                        params[batch_idx, lane_idx],
                        groups=self.groups,
                    ) for lane_idx in range(N)
                ]))
        return torch.stack(batch_tensors)


class HeadLanePredict(BaseModule):

    def __init__(self,
                 in_dim,
                 img_h,
                 img_w,
                 dropout=0.2,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='ReLU'),
                 init_cfg=None):
        super(HeadLanePredict, self).__init__(init_cfg)
        self.img_h, self.img_w = img_h, img_w
        self.in_dim = in_dim

        def up_layer(in_dim, out_dim):
            return nn.Sequential(
                ConvTranspose2d(in_dim,
                                out_dim,
                                kernel_size=3,
                                stride=2,
                                padding=1,
                                output_padding=1),
                nn.BatchNorm2d(out_dim),
                nn.LeakyReLU(0.1, inplace=True),
            )

        self.lane_conv = nn.Sequential(
            up_layer(in_dim, in_dim // 4),
            nn.Upsample(size=(self.img_h, self.img_w), mode="bilinear", align_corners=True),
            ConvModule(in_dim // 4,
                       1,
                       kernel_size=3,
                       stride=1,
                       padding=1,
                       act_cfg=None,
                       padding_mode='reflect'),
        )  # B, 1, 360, 640

        # Is there a lane for each line
        self.lane_linear = Sequential(
            nn.LeakyReLU(0.1, inplace=True),
            Linear(self.img_w, 64, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            Linear(64, 2, bias=True),
        )  # B, 1, 360, 1

    def forward(self, ins_feat):
        lane = self.lane_conv(ins_feat)  # B, 1, 360, 640
        idx_lane = self.lane_linear(lane)  # B, 1, 360, 2
        idx_lane = idx_lane.squeeze(1)  # B, 360, 2
        return lane, idx_lane


class HoughUpsampleBlock(BaseModule):

    def __init__(self,
                 in_dim,
                 out_dim,
                 kernel_size=3,
                 stride=2,
                 padding=1,
                 output_padding=1,
                 init_cfg=None):
        super().__init__(init_cfg)
        self.up_conv = nn.Sequential(
            ConvTranspose2d(
                in_dim,
                out_dim,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                output_padding=output_padding,
            ),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),  # nn.Mish(inplace=True),
        )
        self.gp_mlp = nn.Sequential(
            nn.Linear(out_dim * 2, out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(out_dim, out_dim),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.up_conv(x)
        x_avg = F.adaptive_avg_pool2d(x, (1, 1))  # [B, C, 1, 1]
        x_max = F.adaptive_max_pool2d(x, (1, 1))  # [B, C, 1, 1]
        x_gp = torch.cat([x_avg, x_max], dim=1)  # [B, C*2, 1, 1]
        x_gp = self.gp_mlp(x_gp.flatten(1)).reshape_as(x_avg)  # [B, C, 1, 1]
        x = x + x * x_gp  # residual self-gating
        return x


class HoughPointHead(BaseModule):

    def __init__(self, in_dim, num_angle, num_rho, init_cfg=None):
        super().__init__(init_cfg)
        self.upsample = nn.Upsample(
            size=(num_angle, num_rho),
            mode="bilinear",
            align_corners=True,
        )
        self.final_conv = ConvModule(in_dim,
                                     1,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1,
                                     norm_cfg=None,
                                     act_cfg=None)

    def forward(self, x):
        x = self.upsample(x)
        x = self.final_conv(x)
        return x


class HoughOffsetHead(BaseModule):

    def __init__(self, in_dim, img_h, img_w, init_cfg=None):
        super().__init__(init_cfg)

        def up_layer(in_dim, out_dim):
            return nn.Sequential(
                ConvTranspose2d(in_dim,
                                out_dim,
                                kernel_size=3,
                                stride=2,
                                padding=1,
                                output_padding=1),
                nn.BatchNorm2d(out_dim),
                nn.LeakyReLU(0.1, inplace=True),
            )

        self.up_blocks = nn.Sequential(
            up_layer(in_dim, in_dim // 2),
            up_layer(in_dim // 2, in_dim // 4),
        )

        self.upsample = nn.Upsample(
            size=(img_h, img_w),
            mode="bilinear",
            align_corners=True,
        )
        self.final_conv = ConvModule(in_dim // 4,
                                     2,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1,
                                     act_cfg=None,
                                     padding_mode='reflect')

    def forward(self, x):
        x = self.up_blocks(x)
        x = self.upsample(x)
        x = self.final_conv(x)
        x = torch.tanh(x)  # [-1, 1]
        return x


class HoughLineHead(BaseModule):

    def __init__(self, in_dim, img_h, img_w, init_cfg=None):
        super().__init__(init_cfg)

        def up_layer(in_dim, out_dim):
            return nn.Sequential(
                ConvTranspose2d(in_dim,
                                out_dim,
                                kernel_size=3,
                                stride=2,
                                padding=1,
                                output_padding=1),
                nn.BatchNorm2d(out_dim),
                nn.LeakyReLU(0.1, inplace=True),
            )

        self.up_blocks = nn.Sequential(
            up_layer(in_dim, in_dim // 4), )

        self.upsample = nn.Upsample(
            size=(img_h, img_w),
            mode="bilinear",
            align_corners=True,
        )
        self.final_conv = ConvModule(in_dim // 4,
                                     1,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1,
                                     act_cfg=None,
                                     padding_mode='reflect')

    def forward(self, x):
        x = self.up_blocks(x)
        x = self.upsample(x)
        x = self.final_conv(x)
        return x


class HoughNet(BaseModule):

    def __init__(self,
                 in_dim,
                 param_dim,
                 max_num_lane,
                 num_angle,
                 num_rho,
                 hough_scale,
                 threshold=0.1,
                 nms_kernel_size=5,
                 select_mode='nms',
                 init_cfg=None):
        super().__init__(init_cfg)
        self.num_angle = num_angle
        self.num_rho = num_rho
        self.hough_scale = hough_scale
        self.select_mode = select_mode

        self.point_head = HoughPointHead(in_dim, num_angle, num_rho)
        self.param_head = nn.Sequential(
            Linear(in_dim, in_dim * 2, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            Linear(in_dim * 2, param_dim, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
        )  # param_dim = self.out_dim * (self.in_dim // self.groups)

        # learn the hough feature of non-lane
        self.fea_no_lane = Parameter(torch.zeros(in_dim), True)  # TODO: find a feature for no lane
        self.max_num_lane = max_num_lane
        self.threshold = threshold
        self.nms_kernel_size = nms_kernel_size

    def _find_points(self, point_map):
        threshold = torch.quantile(point_map.flatten(), q=0.999, dim=0)  # Adaptive threshold
        threshold = max(self.threshold, threshold)

        binary_point_map = point_map > threshold  # [B, 1, Angle, Rho]
        binary_point_map = binary_point_map.squeeze().cpu().numpy()
        point_map = rearrange(point_map, 'c h w -> h w c').detach().cpu().numpy()

        label_map = label(binary_point_map, connectivity=1)
        props = regionprops(label_map, intensity_image=point_map)
        props.sort(key=lambda p: p.intensity_mean, reverse=True)
        props = props[:self.max_num_lane]
        props.sort(key=lambda p: p.centroid[0] * 1e4 + p.centroid[1])
        points = [prop.centroid for prop in props]
        return points

    def _point_to_index(self, point):
        angle = max(min(round(point[0]), self.num_angle - 1), 0)
        rho = max(min(round(point[1]), self.num_rho - 1), 0)
        return angle, rho

    def _get_point_list(self, point_map):
        selection_point_map = torch.sigmoid(point_map)
        points_list = [self._find_points(map) for map in selection_point_map]
        points_list = [[self._point_to_index(p) for p in points]
                       for points in points_list]  # [B, NumLanes]
        return points_list

    def _get_point_list_nms(self, point_map):
        # 5, 11, 17, 21
        def _nms(heat, threshold=0.0, kernel=5):
            hmax = F.max_pool2d(heat, (kernel, kernel),
                                stride=1,
                                padding=(kernel - 1) // 2,
                                return_indices=False)
            keep = ((hmax == heat) & (hmax > threshold)).float()
            return heat * keep

        selection_point_map = torch.sigmoid(point_map)
        selmap_nms = _nms(selection_point_map, threshold=self.threshold, kernel=self.nms_kernel_size)  # [B, 1, H, W]
        selmap_nms = torch.squeeze(selmap_nms, 1)  # [B, H, W]
        points_list = []  # [B, NumLanes]
        for batch_idx in range(selmap_nms.shape[0]):
            values, inds = torch.topk(selmap_nms[batch_idx].flatten(), k=self.max_num_lane, sorted=False)
            points_list.append([
                self._point_to_index((idx.item() // self.num_rho, idx.item() % self.num_rho))
                for v, idx in zip(values, inds) if v > 0.0
            ])

        return points_list

    def forward(self, fea, gt_points_list=None):
        B, D, H, W = fea.shape
        point_map = self.point_head(fea)  # [B, 1, Angle, Rho]

        # get points list
        if gt_points_list is None:
            # Testing stage: Get param indices from hough points
            if self.select_mode == 'nms':
                points_list = self._get_point_list_nms(point_map)
            else:
                points_list = self._get_point_list(point_map)
        else:
            # convert tensor to tuple
            points_list = [[tuple(point.cpu().numpy()) for point in batch_points if point[0] >= 0]
                           for batch_points in gt_points_list]

        # Extract parameters from indices [B, NumLanes, P]
        feas_list = []
        for batch_idx, indices in enumerate(points_list):
            feas_with_lane = []
            for (angle, rho) in indices:
                re_x = int(angle / self.hough_scale)
                re_y = int(rho / self.hough_scale)

                feas_with_lane.append(fea[batch_idx, :, re_x, re_y])
            feas_no_lane = [self.fea_no_lane] * (self.max_num_lane - len(feas_with_lane))
            feas_list.append(torch.stack(feas_with_lane + feas_no_lane, dim=0))  # [max_num_lane, D]

        feas_list = torch.stack(feas_list, dim=0)  # [B, max_num_lane, D]
        params = self.param_head(feas_list)  # [B, max_num_lane, P]
        return point_map, params, points_list


@HEADS.register_module
class DenseLaneHead(BaseModule):

    def __init__(self, image_size, max_num_lane, d_model, d_ins, groups, hough_scale, num_angle, num_rho, fea_size,
                 threshold=0.1, nms_kernel_size=5, d_hough=None, select_mode='nms',
                 train_cfg=None, test_cfg=None, init_cfg=None):
        super(DenseLaneHead, self).__init__(init_cfg)

        self.h_img, self.w_img = to_2tuple(image_size)
        self.max_num_lane = max_num_lane
        self.num_angle = num_angle
        self.num_rho = num_rho
        self.hough_scale = hough_scale  # 6: 360->60
        self.groups = groups
        self.d_model = d_model
        self.d_hough = d_model if d_hough is None else d_hough

        # aggregation feature
        self.dht2 = DHTLayer(in_dim=d_model,
                             out_dim=self.d_hough,
                             num_angle=num_angle // self.hough_scale,
                             num_rho=num_rho // self.hough_scale,
                             init_cfg=init_cfg)
        self.dht3 = DHTLayer(in_dim=d_model,
                             out_dim=self.d_hough,
                             num_angle=num_angle // self.hough_scale // 2,
                             num_rho=num_rho // self.hough_scale // 2,
                             init_cfg=init_cfg)
        self.dht4 = DHTLayer(in_dim=d_model,
                             out_dim=self.d_hough,
                             num_angle=num_angle // self.hough_scale // 4,
                             num_rho=num_rho // self.hough_scale // 4,
                             init_cfg=init_cfg)
        self.fea_conv = ConvModule(
            self.d_hough * 3,
            d_model,
            kernel_size=1,
            stride=1,
            norm_cfg=dict(type='BN', requires_grad=True),
            act_cfg=dict(type='ReLU'),
        )

        # reverse aggregation feature
        fea_height, fea_width = to_2tuple(fea_size)
        self.rht_hough = RHTLayer(in_dim=d_model,
                                  out_dim=d_model,
                                  img_height=fea_height,
                                  img_width=fea_width,
                                  init_cfg=init_cfg)
        self.hough_line_head = HoughLineHead(in_dim=d_model, img_h=self.h_img, img_w=self.w_img, init_cfg=init_cfg)

        # lane detection head
        self.ins_head = InstanceHead(d_model, d_ins, groups=groups, init_cfg=init_cfg)
        self.pre_lanes = HeadLanePredict(in_dim=d_ins,
                                         img_h=self.h_img,
                                         img_w=self.w_img,
                                         init_cfg=init_cfg)

        # feature aggregation head
        self.hough_net = HoughNet(in_dim=d_model,
                                  param_dim=self.ins_head.dim_param,
                                  max_num_lane=max_num_lane,
                                  num_angle=num_angle,
                                  num_rho=num_rho,
                                  hough_scale=self.hough_scale,
                                  threshold=threshold,
                                  nms_kernel_size=nms_kernel_size,
                                  select_mode=select_mode,
                                  init_cfg=init_cfg)

    def feature_fusion(self, *feats, size=None, mode='bilinear'):
        if size is None:
            angle, rho = self.num_angle // self.hough_scale, self.num_rho // self.hough_scale
            size = (angle, rho)

        feats = [F.interpolate(f, size=size, mode=mode) for f in feats]
        fusion_feats = torch.cat(feats, dim=1)
        fusion_feats = self.fea_conv(fusion_feats)  # B, 128, 120, 72
        return fusion_feats

    def _forward_base(self, x, mode, gt_points_list=None):
        feat_1, feat_2, feat_3, feat_4 = x

        # [B, D, Angle, Rho]
        hough_feat4 = self.dht4(feat_4)  # torch.Size([2, 128, 12, 20])
        hough_feat3 = self.dht3(feat_3)  # torch.Size([2, 128, 23, 40])
        hough_feat2 = self.dht2(feat_2)  # torch.Size([2, 128, 45, 80])
        hough_feat = self.feature_fusion(hough_feat2, hough_feat3, hough_feat4)  # [B, 128*3, Angle, Rho]

        # [B, 1, Angle, Rho], [B, P, Angle, Rho]
        point_map, params, points_list = self.hough_net(hough_feat, gt_points_list)

        ins_feats = self.ins_head(feat_1, params)  # [B, max_num_lane, out_dim, 36, 64]
        ins_feats = rearrange(ins_feats, 'b n c h w -> (b n) c h w')
        lane_maps, idx_lanes = self.pre_lanes(ins_feats)
        lane_maps = rearrange(lane_maps, '(b n) c h w -> b (n c) h w', c=1, n=self.max_num_lane)
        idx_lanes = rearrange(idx_lanes, '(b n) h w -> b n h w', n=self.max_num_lane)

        if mode == 'infer':
            return lane_maps, idx_lanes

        # for train & test
        re_hough_feat = self.rht_hough(hough_feat)
        line_map = self.hough_line_head(re_hough_feat)

        return lane_maps, idx_lanes, point_map, points_list, line_map, feat_1

    def forward_train(self, x, gt_points_list):
        return self._forward_base(x, 'train', gt_points_list)

    def forward_test(self, x, gt_points_list=None):
        return self._forward_base(x, 'test', gt_points_list)

    def forward_inference(self, x):
        return self._forward_base(x, 'infer')
