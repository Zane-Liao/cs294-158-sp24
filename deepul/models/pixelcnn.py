import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union

from deepul.models.modules.layers import MaskedConv2d, Residualblock

__all__ = {
    "SimplePixelCNN",
    "PixelCNN",
}

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
    def __init__(self, in_channels: int = 1, n_filters: int = 64):
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
        x = x.float()
        logits = self.forward(x)
        return F.binary_cross_entropy_with_logits(logits, x, reduction="mean")
    
    def sample(self, n_samples: int = 100, image_size: int = 20) -> torch.Tensor:
        self.eval()
        
        x = torch.zeros(n_samples, 1, image_size, image_size)
        
        for i in range(image_size):
            for j in range(image_size):
                logits = self.forward(x)
                p = torch.sigmoid(logits[:, :, i, j])
                x[:, :, i, j] = torch.bernoulli(p)

        return x


class PixelCNN(nn.Module):
    def __init__(self,
                 in_channels: int = 3,
                 n_filters: int = 128,
                 n_res_blocks: int = 8,
                 n_classes_per_channel: int = 4
                 ):
        super().__init__()
        pad_7 = 7 // 2
        
        self.conv_in = MaskedConv2d(
            'A', in_channels, n_filters, kernel_size=7, padding=pad_7, channel_conditioning=None
        )
        
        self.res_block = nn.Sequential(*[Residualblock(in_channels, kernel_size=7, _layer_norm=True) for _ in range(n_res_blocks)])
        
        self.post_relu = nn.ReLU(inplace=True)
        
        self.post_conv_1 = nn.Conv2d(n_filters, n_filters, kernel_size=1)
        self.post_conv_2 = nn.Conv2d(n_filters, n_filters, kernel_size=1)
        
        self.final = nn.Conv2d(n_filters, in_channels*n_classes_per_channel, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        out = self.post_relu(self.res_block(self.conv_in(x)))
        
        out = self.post_relu(self.post_conv_1(out))
        
        return self.final(self.post_conv_2(out))
    
    def loss(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.forward(x)
        return F.cross_entropy(logits, x)
    
    def sample(self, batch_size, image_size) -> torch.Tensor:
        
        raise NotImplementedError