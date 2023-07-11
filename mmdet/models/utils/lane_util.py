import math
import torch
from mmcv.runner import BaseModule
from mmcv.cnn import ConvModule
from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence


class PositionalEmbedding(BaseModule):
    def __init__(self, d_model, max_len=256):
        super().__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x, h=None):
        if h is not None:
            return self.pe[:, :h]
        return self.pe[:, :x.size(1)]  # (1, max_len, d_model)


class VitEncoder(BaseModule):
    def __init__(self, d_model, num_heads, num_layers, seq_len, drop_rate=0.1,
                 act_cfg=dict(type='GELU'), norm_cfg=dict(type='LN'), init_cfg=None):
        super(VitEncoder, self).__init__(init_cfg)

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
        x = self.vit_encoder.forward(query=x, key=None, value=None, query_pos=pos, key_pos=pos)
        return x



class ResBlock(BaseModule):
    """Basic residual convolution block."""
    def __init__(self, 
                 in_channels, 
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='LeakyReLU', negative_slope=0.1),
                 init_cfg=None):
        super(ResBlock, self).__init__(init_cfg)

        cfg = dict(conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)

        self.conv1 = ConvModule(in_channels, in_channels, 3, padding=1, **cfg)
        self.conv2 = ConvModule(in_channels, in_channels, 3, padding=1, **cfg)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + residual

        return out

