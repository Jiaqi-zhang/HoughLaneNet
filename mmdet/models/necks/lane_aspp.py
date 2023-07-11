import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule, auto_fp16

from ..builder import NECKS
from ..utils.lane_util import ResBlock


@NECKS.register_module()
class ASPP(BaseModule):
    def __init__(self,
                 in_channels,
                 out_channels,
                 output_stride,
                 extend_channels,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='ReLU'),
                 upsample_cfg=dict(mode='bilinear', align_corners=True),
                 init_cfg=None):
        super(ASPP, self).__init__(init_cfg)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.output_stride = output_stride
        self.fp16_enabled = False
        self.upsample_cfg = upsample_cfg.copy()

        if output_stride == 16:
            dilations = [1, 6, 12, 18]
        elif output_stride == 8:
            dilations = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        self.aspp1 = ConvModule(self.in_channels, self.out_channels, kernel_size=1, stride=1, padding=0,
                                dilation=dilations[0], norm_cfg=norm_cfg, act_cfg=act_cfg, padding_mode='reflect')
        self.aspp2 = ConvModule(self.in_channels, self.out_channels, kernel_size=3, stride=1, padding=dilations[1],
                                dilation=dilations[1], norm_cfg=norm_cfg, act_cfg=act_cfg, padding_mode='reflect')
        self.aspp3 = ConvModule(self.in_channels, self.out_channels, kernel_size=3, stride=1, padding=dilations[2],
                                dilation=dilations[2], norm_cfg=norm_cfg, act_cfg=act_cfg, padding_mode='reflect')
        self.aspp4 = ConvModule(self.in_channels, self.out_channels, kernel_size=3, stride=1, padding=dilations[3],
                                dilation=dilations[3], norm_cfg=norm_cfg, act_cfg=act_cfg, padding_mode='reflect')

        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            ConvModule(self.in_channels, self.out_channels, kernel_size=1, stride=1, norm_cfg=norm_cfg, act_cfg=act_cfg)
        )
        self.conv_1x1_output = ConvModule(self.out_channels * 5, self.out_channels, kernel_size=1, stride=1,
                                          norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.dropout = nn.Dropout(0.5)  # DeepLabV3 default

        # last feature to rescale
        self.fea_inter_conv = ConvModule(self.in_channels, self.in_channels, kernel_size=1, stride=1,
                                         norm_cfg=norm_cfg, act_cfg=act_cfg)

        # DeepLabV3 + extends
        self.fea_conv = ConvModule(extend_channels, self.out_channels, kernel_size=1, stride=1,
                                   norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.fea_out_conv = ConvModule(self.out_channels * 2, self.out_channels, kernel_size=3, stride=1, padding=1,
                                       norm_cfg=norm_cfg, act_cfg=act_cfg, padding_mode='reflect')

        self.resnet = nn.Sequential(*[ResBlock(self.out_channels) for _ in range(4)])

    @auto_fp16()
    def forward(self, x):
        """Forward function."""

        last_fea = x[-1]  # [2, 256, 12, 20]

        # TODO ResNet=-3 | PVT=-2
        sel_fea = x[-3]  # [2, 160, 45, 80]

        last_fea = F.interpolate(last_fea, size=(36, 64), mode=self.upsample_cfg['mode'],
                                 align_corners=self.upsample_cfg['align_corners'])
        last_fea = self.fea_inter_conv(last_fea)

        # aspp
        x1 = self.aspp1(last_fea)
        x2 = self.aspp2(last_fea)
        x3 = self.aspp3(last_fea)
        x4 = self.aspp4(last_fea)
        x5 = self.global_avg_pool(last_fea)
        x5 = F.interpolate(x5, size=x4.shape[2:], mode=self.upsample_cfg['mode'],
                           align_corners=self.upsample_cfg['align_corners'])
        last_fea = torch.cat((x1, x2, x3, x4, x5), dim=1)
        last_fea = self.conv_1x1_output(last_fea)
        last_fea = self.dropout(last_fea)

        # DeepLabV3+extends
        last_fea = F.interpolate(last_fea, size=sel_fea.shape[2:], mode=self.upsample_cfg['mode'],
                                 align_corners=self.upsample_cfg['align_corners'])
        sel_fea = self.fea_conv(sel_fea)
        out = torch.cat((last_fea, sel_fea), dim=1)  # [2, d_model*2, 45, 80]
        out = self.fea_out_conv(out)  # [2, d_model, 45, 80]

        # argument feature
        out = self.resnet(out)
        return out
