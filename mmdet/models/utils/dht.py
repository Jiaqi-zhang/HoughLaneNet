import torch
from torch import nn
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule
from .dht_op.dht_func import DHTFunction, RHTFunction


class DHT(BaseModule):

    def __init__(self, num_angle, num_rho):
        super(DHT, self).__init__()
        self.num_angle = num_angle
        self.num_rho = num_rho

    def forward(self, feat):
        return DHTFunction.apply(feat, self.num_angle, self.num_rho)


class DHTLayer(BaseModule):

    def __init__(self,
                 in_dim,
                 out_dim,
                 num_angle,
                 num_rho,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='ReLU'),
                 init_cfg=None):
        super(DHTLayer, self).__init__(init_cfg)

        self.in_conv = ConvModule(in_dim, out_dim, 1, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.dht = DHT(num_angle, num_rho)
        self.out_conv = nn.Sequential(
            ConvModule(out_dim, out_dim, 3, padding=1, norm_cfg=norm_cfg, act_cfg=act_cfg, padding_mode='reflect'),
            ConvModule(out_dim, out_dim, 3, padding=1, norm_cfg=norm_cfg, act_cfg=act_cfg, padding_mode='reflect'),
        )

    def forward(self, x):
        x = self.in_conv(x)
        x = self.dht(x)
        x = self.out_conv(x)
        return x


class RHT(BaseModule):

    def __init__(self, img_height, img_width):
        super(RHT, self).__init__()
        self.img_height = img_height
        self.img_width = img_width

    def forward(self, feat):
        return RHTFunction.apply(feat, self.img_height, self.img_width)


class RHTLayer(BaseModule):

    def __init__(self,
                 in_dim,
                 out_dim,
                 img_height,
                 img_width,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='ReLU'),
                 init_cfg=None):
        super(RHTLayer, self).__init__(init_cfg)

        self.in_conv = ConvModule(in_dim, out_dim, 1, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.rht = RHT(img_height, img_width)
        self.out_conv = nn.Sequential(
            ConvModule(out_dim, out_dim, 3, padding=1, norm_cfg=norm_cfg, act_cfg=act_cfg, padding_mode='reflect'),
            ConvModule(out_dim, out_dim, 3, padding=1, norm_cfg=norm_cfg, act_cfg=act_cfg, padding_mode='reflect'),
        )

    def forward(self, x):
        x = self.in_conv(x)
        x = self.rht(x)
        x = self.out_conv(x)
        return x