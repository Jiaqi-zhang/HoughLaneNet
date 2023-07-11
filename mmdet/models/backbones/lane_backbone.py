import warnings
from collections import OrderedDict
from copy import deepcopy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from mmcv.cnn import build_norm_layer, constant_init, trunc_normal_init, kaiming_init
from mmcv.cnn.bricks.transformer import build_dropout
from mmcv.runner import BaseModule, _load_checkpoint
from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence

from einops import rearrange
from ...utils import get_root_logger
from ..builder import BACKBONES
from ..utils.transformer import PatchEmbed
from ..utils.lane_util import PositionalEmbedding


class VitTransformerEncoder(BaseModule):
    def __init__(self,
                 d_model,
                 num_heads,
                 num_layers,
                 seq_len,
                 drop_rate=0.1,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 init_cfg=None):
        super(VitTransformerEncoder, self).__init__(init_cfg)

        self.pos_emb = PositionalEmbedding(d_model, max_len=seq_len)

        transformerlayers = dict(
            type='TransformerLayerSequence',
            transformerlayers=dict(
                type='BaseTransformerLayer',
                attn_cfgs=dict(
                    type='MultiheadAttention',
                    embed_dims=d_model,
                    num_heads=num_heads,
                    dropout_layer=dict(type='Dropout', drop_prob=drop_rate),
                    batch_first=True,
                ),
                ffn_cfgs=dict(
                    type='FFN',
                    embed_dims=d_model,
                    feedforward_channels=d_model * 4,
                    num_fcs=2,
                    ffn_drop=0.1,
                    act_cfg=act_cfg,
                ),
                operation_order=('self_attn', 'norm', 'ffn', 'norm'),
                norm_cfg=norm_cfg,
                batch_first=True,
            ),
            num_layers=num_layers,
        )
        self.vit_encoder = build_transformer_layer_sequence(cfg=transformerlayers)

    def forward(self, x):
        pos = self.pos_emb(x)
        emb_data = self.vit_encoder.forward(query=x, key=None, value=None, query_pos=pos, key_pos=pos)
        return emb_data


@BACKBONES.register_module()
class LaneTransformer(BaseModule):
    def __init__(self,
                 in_channels=3,
                 embed_dims=96,
                 img_size=(360, 640),
                 patch_size=10,
                 num_layers=6,
                 num_heads=3,
                 drop_rate=0.1,
                 patch_norm=True,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 pretrained=None,
                 init_cfg=None):

        # Load pretrained weights
        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be specified at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is None:
            self.init_cfg = init_cfg
        else:
            raise TypeError('pretrained must be a str or None')
        super(LaneTransformer, self).__init__(init_cfg=init_cfg)
        self.img_h, self.img_w = img_size
        self.patch_size = patch_size

        self.patch_embed = PatchEmbed(
            in_channels=in_channels,
            embed_dims=embed_dims,
            conv_type='Conv2d',
            kernel_size=patch_size,
            stride=patch_size,
            norm_cfg=norm_cfg if patch_norm else None,
            init_cfg=None)

        self.seq_len = (img_size[0] // patch_size) * (img_size[1] // patch_size)
        self.vit_encoder = VitTransformerEncoder(d_model=embed_dims, num_heads=num_heads, num_layers=num_layers,
                                                 seq_len=self.seq_len, drop_rate=drop_rate,
                                                 act_cfg=act_cfg, norm_cfg=norm_cfg)

    def init_weights(self):
        logger = get_root_logger()
        if self.init_cfg is None:
            logger.warn(f'No pre-trained weights for '
                        f'{self.__class__.__name__}, '
                        f'training start from scratch')

            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, nn.Linear):
                    trunc_normal_init(m, std=.02, bias=0.)
                elif isinstance(m, nn.LayerNorm):
                    constant_init(m, 1.0)
        else:
            assert 'checkpoint' in self.init_cfg, f'Only support ' \
                                                  f'specify `Pretrained` in ' \
                                                  f'`init_cfg` in ' \
                                                  f'{self.__class__.__name__} '
            ckpt = _load_checkpoint(self.init_cfg.checkpoint, logger=logger, map_location='cpu')
            if 'state_dict' in ckpt:
                _state_dict = ckpt['state_dict']
            elif 'model' in ckpt:
                _state_dict = ckpt['model']
            else:
                _state_dict = ckpt

            state_dict = OrderedDict()
            for k, v in _state_dict.items():
                if k.startswith('backbone.'):
                    state_dict[k[9:]] = v

            # strip prefix of state_dict
            if list(state_dict.keys())[0].startswith('module.'):
                state_dict = {k[7:]: v for k, v in state_dict.items()}

            # load state_dict
            self.load_state_dict(state_dict, False)

    def forward(self, x):
        x, hw_shape = self.patch_embed(x)
        out = self.vit_encoder(x)
        out = rearrange(out, 'b (h w) c -> b c h w', h=(self.img_h // self.patch_size))
        # print(out.shape)
        return out
