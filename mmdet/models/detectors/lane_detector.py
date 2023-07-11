import torch
from torch import nn

from mmcv.cnn import ConvModule, ConvTranspose2d
from mmcv.runner.base_module import BaseModule

from .single_stage import SingleStageDetector
from ..builder import DETECTORS
from ..losses import LaneLoss


class NeckMapPredict(BaseModule):

    def __init__(self, in_dim, img_h, img_w, act_sigmoid=False, init_cfg=None):
        super(NeckMapPredict, self).__init__(init_cfg)
        self.img_h, self.img_w = img_h, img_w

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

        self.conv = nn.Sequential(
            up_layer(in_dim, in_dim // 4),
            nn.Upsample(size=(self.img_h, self.img_w), mode="bilinear", align_corners=True),
            ConvModule(in_dim // 4,
                       1,
                       kernel_size=3,
                       stride=1,
                       padding=1,
                       act_cfg=None,
                       padding_mode='reflect'),
        )
        self.activation = nn.Sigmoid() if act_sigmoid else None

    def forward(self, x):
        out = self.conv(x)
        if self.activation:
            out = self.activation(out)
        return out


@DETECTORS.register_module
class LaneDetector(SingleStageDetector):

    def __init__(self,
                 backbone,
                 base_data,
                 neck,
                 head,
                 loss_weights,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(LaneDetector, self).__init__(backbone=backbone,
                                           neck=neck,
                                           bbox_head=head,
                                           train_cfg=train_cfg,
                                           test_cfg=test_cfg,
                                           pretrained=pretrained,
                                           init_cfg=init_cfg)
        # Note: here is the assignment of dict, the model will only be created after the base class is built

        self.img_h = base_data['image_height']
        self.img_w = base_data['image_width']
        self.patch_h = base_data['patch_height']
        self.patch_w = base_data['patch_width']
        self.d_model = base_data['d_model']

        # map head
        self.pre_seg_map = NeckMapPredict(in_dim=self.d_model, img_h=self.img_h, img_w=self.img_w)

        self.loss = LaneLoss(img_w=self.img_w, weights=loss_weights)

    def forward(self, img, img_metas=None, return_loss=True, training_test=True, **kwargs):
        """
        Calls either forward_train or forward_test depending on whether
        return_loss=True. Note this setting will change the expected inputs.
        When `return_loss=True`, img and img_meta are single-nested (i.e.
        Tensor and List[dict]), and when `resturn_loss=False`, img and img_meta
        should be double nested (i.e.  List[Tensor], List[List[dict]]), with
        the outer list indicating test time augmentations.
        """
        # return self.forward_test(img, img_metas, **kwargs)

        if img_metas is None:
            return self.test_inference(img)
        elif return_loss:
            return self.forward_train(img, img_metas, **kwargs)
        else:
            if training_test:
                return self.forward_test(img, img_metas, **kwargs)
            else:
                return self.test_inference(img)

    def test_inference(self, img):
        backbone_feat = self.backbone(img)  # B, 192, 36, 64
        neck_feat = self.neck(backbone_feat)
        lane_map, idx_lanes = self.bbox_head.forward_inference(neck_feat)
        return [lane_map, idx_lanes]

    def forward_train(self, img, img_metas, **kwargs):
        backbone_feat = self.backbone(img)
        neck_feat = self.neck(backbone_feat)  # [B, d_model, 45, 80]

        lane_map, idx_lanes, pre_hmap, _, pre_lmap, top_feat = \
            self.bbox_head.forward_train(neck_feat, kwargs["point_list"])

        # display mid images
        pre_smap = self.pre_seg_map(top_feat)  # [2, 180, 90, 160] -> B, 1, 360, 640
        losses = self.loss(lane_map, idx_lanes, pre_smap, pre_hmap, pre_lmap, img_metas, **kwargs)
        return losses

    def forward_test(self, img, img_metas, **kwargs):
        """Test without augmentation."""
        backbone_feat = self.backbone(img)  # B, 192, 36, 64
        neck_feat = self.neck(backbone_feat)

        lane_map, idx_lanes, pre_hmap, pre_plist, pre_lmap, top_feat = \
            self.bbox_head.forward_test(neck_feat, kwargs["point_list"])
        pre_smap = self.pre_seg_map(top_feat)  # B, 1, 360, 640

        return [
            lane_map, idx_lanes, pre_smap, pre_hmap, pre_plist, img, kwargs["segment_map"],
            kwargs["hough_map"], kwargs["point_list"], kwargs["line_map"], pre_lmap
        ]

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/analysis_tools/get_flops.py`
        """
        backbone_feat = self.backbone(img)  # B, 192, 36, 64
        neck_feat = self.neck(backbone_feat)
        lane_map, idx_lanes = self.bbox_head.forward_inference(neck_feat)
        return [lane_map, idx_lanes]
