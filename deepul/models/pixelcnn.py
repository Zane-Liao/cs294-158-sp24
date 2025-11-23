import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union

__all__ = {
    "SimplePixelCNN",
    # "PixelCNN",
}

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
        out = self.conv(self.relu(x))
        
        if self.layer_norm is not None:
            # x: (B, C, H, W) -> permute to (B, H, W, C)
            out = out.permute(0, 2, 3, 1)
            out = self.layer_norm(out)
            out = out.permute(0, 3, 1, 2)
        
        out = self.relu(out)
        out = self.conv1x1(out)
        
        return x + out


class SimplePixelCNN(nn.Module):
    """
        A simple PixelCNN architecture to model binary MNIST and shapes images
        (same as Q2(b), but with a PixelCNN).
        
        * A $7 \times 7$ masked type A convolution
        * $5$ $7 \times 7$ masked type B convolutions
        * $2$ $1 \times 1$ masked type B convolutions
        * Appropriate ReLU nonlinearities in-between
        * 64 convolutional filters
    """
    def __init__(self, in_channels: int = 3, n_filters: int = 64):
        super().__init__()
        pad_7 = 7 // 2
        
        self.conv_in = MaskedConv2d(
            'A', in_channels, n_filters, kernel_size=7, padding=pad_7, channel_conditioning=None
        )
        
        blocks = []
        for _ in range(5):
            blocks.append(nn.ReLU(inplace=True))
            blocks.append(MaskedConv2d('B', n_filters, n_filters, kernel_size=7, padding=pad_7, channel_conditioning=None))
            
        blocks.append(nn.ReLU(inplace=True))
        blocks.append(nn.Conv2d(n_filters, n_filters, kernel_size=1))
        
        blocks.append(nn.ReLU(inplace=True))
        blocks.append(nn.Conv2d(n_filters, n_filters, kernel_size=1))
        
        self.body = nn.Sequential(*blocks)
        
        self.relu = nn.ReLU(inplace=True)
        
        self.final = nn.Conv2d(n_filters, 1, kernel_size=1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.final(self.relu(self.body(self.conv_in(x))))
    
    def loss(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.forward(x)
        return F.binary_cross_entropy_with_logits(logits, x, reduction="mean")
    
    def samples(self, n_samples: 100, image_size) -> torch.Tensor:
        self.eval()
        
        x = torch.zeros(n_samples, 1, image_size, image_size)
        
        for i in range(image_size):
            for j in range(image_size):
                logits = self.forward(x)
                p = torch.sigmoid(logits[:, :, i, j])
                x[:, :, i, j] = torch.bernoulli(p)

        return x


class PixelCNN(nn.Module):
    def __init__(self):
        super().__init__()
        
        MaskedConv2d(
            
        )
        
        Residualblock(
            
        )
        
        MaskedConv2d(
            
        )
        raise NotImplementedError
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
    
    def loss(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
    
    def samples(self, batch_size, image_size) -> torch.Tensor:
        raise NotImplementedError