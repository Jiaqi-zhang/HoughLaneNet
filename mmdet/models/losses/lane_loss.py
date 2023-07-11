import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from mmdet.models.builder import LOSSES


class FocalLogitsLoss(nn.Module):
    '''nn.Module warpper for focal loss'''

    def __init__(self, gamma=4, weights_list=None):
        super(FocalLogitsLoss, self).__init__()
        self.gamma = gamma
        self.weights_list = weights_list

    def _neg_loss(self, pred, gt, channel_weights=None):
        ''' 
        Modified focal loss. Exactly the same as CornerNet.
        Runs faster and costs a little bit more memory
        Arguments:
        pred (batch x c x h x w)
        gt (batch x c x h x w)
        '''
        pos_inds = gt.ge(1).float()
        neg_inds = gt.lt(1).float()
        neg_weights = torch.pow(1 - gt, self.gamma)  # assume gt is in [0,1]

        sigmoid_pred = torch.sigmoid(pred)
        pos_loss = F.logsigmoid(pred) * torch.pow(1 - sigmoid_pred, 2) * pos_inds
        neg_loss = F.logsigmoid(-pred) * torch.pow(sigmoid_pred, 2) * neg_weights * neg_inds

        num_pos = pos_inds.float().sum()
        if channel_weights is None:
            pos_loss = pos_loss.sum()
            neg_loss = neg_loss.sum()
        else:
            pos_loss_sum = 0
            neg_loss_sum = 0
            for i in range(len(channel_weights)):
                p = pos_loss[:, i, :, :].sum() * channel_weights[i]
                n = neg_loss[:, i, :, :].sum() * channel_weights[i]
                pos_loss_sum += p
                neg_loss_sum += n
            pos_loss = pos_loss_sum
            neg_loss = neg_loss_sum

        if num_pos > 0:
            loss = -(pos_loss + neg_loss) / num_pos
        else:
            loss = -neg_loss
        return loss

    def forward(self, out, target):
        return self._neg_loss(out, target, self.weights_list)


class MutilBCEWithLogitsLoss(nn.Module):

    def __init__(self, pos_weight):
        super(MutilBCEWithLogitsLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))

    def forward(self, input, target):
        input = input.permute(0, 2, 3, 1)
        target = target.permute(0, 2, 3, 1)
        loss = self.bce(input, target)
        return loss


class RowFocalLoss(nn.Module):

    def __init__(self):
        super(RowFocalLoss, self).__init__()

    def forward(self, logits, labels, labels_mask):
        N, C, H, W = logits.shape  # [1, 4, 360, 640]
        logits = F.softmax(logits, dim=3)  # B, 4, 360, 640
        embedding = torch.Tensor(np.arange(W)).float().to(logits.device).view(1, 1, 1, -1)
        logits = torch.sum(logits * embedding, dim=3)  # B, 5, 360

        loss = F.l1_loss(logits * labels_mask, labels * labels_mask, reduction='sum')
        mask = labels_mask.bool().float()
        loss = loss / (mask.sum() + 1e-4)
        return loss


class HoughSplitLoss(nn.Module):

    def __init__(self):
        super(HoughSplitLoss, self).__init__()
        self.reg_th = 10.
        self.radius = 5

    def _point_map_loss(self, pre_point, gt_point, pre_hough, gt_hough, height, width):
        point = pre_point if gt_point[0] < 0 else gt_point

        x, y = int(point[0]), int(point[1])  # angle, rho
        left, right = min(x, self.radius), min(width - x, self.radius + 1)
        top, bottom = min(y, self.radius), min(height - y, self.radius + 1)

        gt_masked_map = gt_hough[x - left:x + right, y - top:y + bottom]
        pre_masked_map = pre_hough[x - left:x + right, y - top:y + bottom]
        loss = F.l1_loss(pre_masked_map, gt_masked_map, reduction='sum')

        if pre_point[0] < 0 or gt_point[0] < 0:
            loss = loss * self.reg_th
        return loss / (self.radius * self.radius)

    def forward(self, pre_hough, gt_hough, in_pre_points, gt_points):
        pre_hough = torch.sigmoid(pre_hough)

        B, max_num_lanes = gt_points.shape[0], gt_points.shape[1]
        Angle, Rho = gt_hough.shape[-2], gt_hough.shape[-1]  # pre_hough: [B, 1, Angle, Rho]

        with torch.no_grad():
            pre_points = [sorted(points, key=lambda p: p[0] * 1e4 + p[1]) for points in in_pre_points]
            pre_points = [points + [[-1, -1]] * (max_num_lanes - len(points))
                          if len(points) < max_num_lanes else points for points in pre_points]

        loss = 0.
        for i in range(B):
            gt_batch_points = gt_points[i]
            pre_batch_points = pre_points[i]

            pre_batch_hough = pre_hough[i, 0]
            gt_batch_hough = gt_hough[i, 0]

            for j, (pre_point, gt_point) in enumerate(zip(pre_batch_points, gt_batch_points)):
                if pre_point[0] < 0 and gt_point[0] < 0:
                    continue
                loss += self._point_map_loss(pre_point, gt_point, pre_batch_hough, gt_batch_hough, Angle, Rho)

        loss = F.l1_loss(pre_hough, gt_hough) if loss == 0 else loss / B
        return loss


@LOSSES.register_module
class LaneLoss(torch.nn.Module):

    def __init__(self, img_w, weights):
        """
        Args:
            weights is a dict which sets the weight of the loss
            eg. {hm_weight: 1, kp_weight: 1, ins_weight: 1}
        """
        super(LaneLoss, self).__init__()
        self.img_w = img_w

        self.crit_bce = nn.BCEWithLogitsLoss()
        self.crit_ce = nn.CrossEntropyLoss()
        self.crit_seg = MutilBCEWithLogitsLoss(pos_weight=weights['pos_weight'])

        self.seg_weight = weights['seg_weight']
        self.hough_weight = weights['hough_weight']
        self.line_weight = weights['line_weight']
        self.range_weight = weights['range_weight']
        self.lane_weight = weights['lane_weight']

    def forward(self, lane_map, idx_lanes, pre_smap, pre_hmap, pre_lmap, meta, **kwargs):

        bb_seg_loss = 0
        if self.seg_weight > 0:
            gt_bce_map = torch.sum(kwargs["segment_map"], dim=1)
            gt_bce_map = (gt_bce_map != 0).float().unsqueeze(1)
            bb_seg_loss += self.crit_bce(pre_smap, gt_bce_map)

        hough_loss = 0
        if self.hough_weight > 0:
            gt_hough_map = kwargs["hough_map"]
            hough_loss += F.binary_cross_entropy_with_logits(pre_hmap, gt_hough_map)

        line_loss = 0
        if self.line_weight > 0:
            line_loss += self.crit_bce(pre_lmap, kwargs["line_map"])

        range_loss, lane_loss = 0, 0
        if self.range_weight > 0:
            # idx_lanes: B, 5, 360, 2 | segment_map: [L, H, W]
            gt_idx_lanes = torch.sum(kwargs["segment_map"], dim=-1)  # B, 5, 360
            gt_idx_lanes = (gt_idx_lanes != 0).long()  # B, 5, 360
            range_loss += self.crit_ce(idx_lanes.permute(0, 3, 1, 2), gt_idx_lanes)

        if self.lane_weight > 0:
            lane_loss += self.crit_seg(lane_map, kwargs['segment_map'])

        # Only non-zero losses are valid, otherwise multi-GPU training will report an error
        losses = {}
        if self.seg_weight:
            losses['bb_seg_loss'] = self.seg_weight * bb_seg_loss
        if self.hough_weight:
            losses['hough_loss'] = self.hough_weight * hough_loss
        if self.line_weight:
            losses['line_loss'] = self.line_weight * line_loss
        if self.range_weight:
            losses['range_loss'] = self.range_weight * range_loss
        if self.lane_weight > 0:
            losses['lane_loss'] = self.lane_weight * lane_loss
        return losses
