import torch
from torch.autograd import Function
from torch.nn import Module
import deep_hough_transform as dht


class DHTFunction(Function):
    @staticmethod
    def forward(ctx, feat : torch.Tensor, num_angle, num_rho):
        N, C, _, _ = feat.size()
        out = torch.zeros(N, C, num_angle, num_rho, dtype=feat.dtype, device=feat.device)
        out = dht.forward(out, feat, num_angle, num_rho)
        outputs = out[0]
        ctx.feat_shape = tuple(feat.shape)
        ctx.num_angle = num_angle
        ctx.num_rho = num_rho
        return outputs

    @staticmethod
    def backward(ctx, grad_output : torch.Tensor):
        feat_shape = ctx.feat_shape
        num_angle = ctx.num_angle
        num_rho = ctx.num_rho
        out = torch.zeros(feat_shape, dtype=grad_output.dtype, device=grad_output.device)
        out = dht.backward(out, grad_output.contiguous(), num_angle, num_rho)
        grad_in = out[0]
        return grad_in, None, None



class RHTFunction(Function):
    @staticmethod
    def forward(ctx, feat : torch.Tensor, img_height, img_width):
        N, C, num_angle, num_rho = feat.size()
        out = torch.zeros(N, C, img_height, img_width, dtype=feat.dtype, device=feat.device)
        out = dht.backward(out, feat, num_angle, num_rho)
        outputs = out[0]
        ctx.feat_shape = tuple(feat.shape)
        ctx.num_angle = num_angle
        ctx.num_rho = num_rho
        return outputs

    @staticmethod
    def backward(ctx, grad_output : torch.Tensor):
        feat_shape = ctx.feat_shape
        num_angle = ctx.num_angle
        num_rho = ctx.num_rho
        out = torch.zeros(feat_shape, dtype=grad_output.dtype, device=grad_output.device)
        out = dht.forward(out, grad_output.contiguous(), num_angle, num_rho)
        grad_in = out[0]
        return grad_in, None, None