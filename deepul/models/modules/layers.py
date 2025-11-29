from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union

__all__ = [
    "MaskedConv2d",
    "Residualblock",
    "RMSNorm",
    "RotaryPositionalEmbedding",
    "SwiGLU",
]

class MaskedConv2d(nn.Conv2d):
    """MaskedConv2d for PixelCNN autoregressive modeling. Impl: ['A', 'B'] """
    def __init__(self,
                 mask_type: str,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, Tuple[int, int]],
                 stride: Union[int, Tuple[int, int]] = 1,
                 padding: Union[int, Tuple[int, int]] = 0,
                 dilation: Union[int, Tuple[int, int]] = 1,
                 groups: int = 1,
                 bias: bool = True,
                 channel_conditioning: Optional[int] = None,
                 ):
        super().__init__(in_channels, out_channels, kernel_size,
                         stride=stride, padding=padding, dilation=dilation,
                         groups=groups, bias=bias)
        
        # Check mask_type
        assert mask_type in ('A', 'B'), "mask_type must be 'A' or 'B'."
        self.mask_type = mask_type
        self.channel_conditioning = channel_conditioning
        
        kH, kW = self.kernel_size
        yc, xc = kH // 2, kW // 2
        
        mask = torch.ones((out_channels, in_channels, kH, kW), dtype=self.weight.dtype)
        mask[:, :, yc+1:, :] = 0.0
        mask[:, :, yc, xc+1:] = 0.0
        
        do_rgb_center = False
        cc = channel_conditioning
        if cc is None:
            if in_channels == 3 and (out_channels % 3 == 0):
                cc = 3
                do_rgb_center = True
        else:
            if (in_channels % cc == 0) and (out_channels % cc == 0):
                do_rgb_center = True
            else:
                do_rgb_center = False
                
        if do_rgb_center and cc is not None:
            # e.g. 3 for RGB
            groups = cc
            in_groups_size = in_channels // groups
            out_groups_size = out_channels // groups
            
            # build group ids (vectorized)
            out_groups = torch.arange(out_channels, device=mask.device) // out_groups_size
            in_groups = torch.arange(in_channels, device=mask.device) // in_groups_size
            
            # broadcasting compare
            out_groups = out_groups.unsqueeze(1) # (out,1)
            in_groups = in_groups.unsqueeze(0) # (1,in)
            
            center_mask = (out_groups > in_groups).to(dtype=mask.dtype)
            if mask_type == 'B':
                center_mask = center_mask + (out_groups == in_groups).to(dtype=mask.dtype)
            
            center_mask = (center_mask > 0).to(dtype=mask.dtype)
            
            mask[:, :, yc, xc] = center_mask
        else:
            if mask_type == 'A':
                mask[:, :, yc, xc] = 0.0
            else:
                mask[:, :, yc, xc] = 1.0
                
        self.register_buffer('mask', mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        weight = self.weight * self.mask
        return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    

class Residualblock(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 7, _layer_norm: bool = True):
        super().__init__()
        pad = kernel_size // 2
        
        self.conv = MaskedConv2d(
            'B', channels, channels, kernel_size=kernel_size, padding=pad, channel_conditioning=None,
        )
        
        self.layer_norm = nn.LayerNorm(channels) if _layer_norm else None
        
        # kernel_size=1, Mask_Type='B', and channel_conditioning=None → are equivalent to nn.Conv2d
        self.conv1x1 = nn.Conv2d(channels, channels, kernel_size=1)
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        out = self.conv(self.relu(x))
        
        if self.layer_norm is not None:
            # x: (B, C, H, W) -> permute to (B, H, W, C)
            out = out.permute(0, 2, 3, 1)
            out = self.layer_norm(out)
            out = out.permute(0, 3, 1, 2)
        
        out = self.relu(out)
        out = self.conv1x1(out)
        
        return x + out


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        """"""
        super().__init__()
        half_dim = d_k // 2
        self.theta = (1. / theta ** (torch.arange(0, half_dim, device=device) / half_dim))
        
        self.max_seq_len = torch.arange(max_seq_len, device=device)
        angle_max_seq_len = torch.einsum("i,j->ij", self.max_seq_len, self.theta)

        self.register_buffer("sin", torch.sin(angle_max_seq_len), persistent=False)
        self.register_buffer("cos", torch.cos(angle_max_seq_len), persistent=False)
    
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """"""
        sin = self.sin[token_positions]
        cos = self.cos[token_positions]
        
        x_even = x[..., 0::2] 
        x_odd = x[..., 1::2]
        
        rotated_even = x_even * cos - x_odd * sin
        rotated_odd = x_even * sin + x_odd * cos
        
        return torch.stack([rotated_even, rotated_odd], dim=-1).flatten(-2)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff

        self.w1 = nn.Linear(d_model, d_ff) 
        self.w2 = nn.Linear(d_ff, d_model)
        self.w3 = nn.Linear(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(silu(self.w1(x)) * self.w3(x))


def silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        """"""
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, **factory_kwargs))
        
    def _norm(self, x):
        return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """"""
        # In torch, x.float() => float32
        output = self._norm(x.float()).type_as(x)
        return output * self.weight