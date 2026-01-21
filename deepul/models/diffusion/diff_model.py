import math
from einops import einsum, rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Iterable, Literal, Optional, Tuple, Union

from jaxtyping import Float, Int

from deepul.models.modules.layers import *
import numpy as np
import numpy.typing as npt
from deepul.hw3_utils.lpips import exists, default, once, print_once


__all__ = [
    "GaussianDiffusion",
    "DiT",
    "timestep_embedding",
    "ddpm_sample",
]


class GaussianDiffusion(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    
class DiT(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class ViT(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class ViTBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class UNet(nn.Module):
    def __init__(
        self,
        dim,
        init_dim = None,
        out_dim = None,
        dim_mults = (1, 2, 4, 8),
        channels = 3,
        self_condition = False,
        learned_variance = False,
        learned_sinusoidal_cond = False,
        random_fourier_features = False,
        learned_sinusoidal_dim = 16,
        sinusoidal_pos_emb_theta = 10000,
        dropout = 0.,
        attn_dim_head = 32,
        attn_heads = 4,
        full_attn = None,
        flash_attn = False,
        ) -> None:
        super().__init__()
        
        # determine dimensions
        
        self.channels = channels
        self.self_condition = self_condition
        input_channels = channels * (2 if self_condition else 1)
        
        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(input_channels, init_dim, 7, padding=3)
        
        time_dim = dim * 4
        
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

def timestep_embedding(timesteps, dim, max_period=10000):
    half = dim // 2
    freqs = np.exp(-np.log(max_period) * np.arange(0, half, dtype=torch.float32) / half)
    args = timesteps[:, None].astype(torch.float32) * freqs[None]
    embedding = torch.cat([np.cos(args), np.sin(args)], axis=-1)
    if dim % 2:
        embedding = torch.cat([embedding, np.zeros_like(embedding[:, :1])], axis=-1)
    return embedding

def ddpm_sample():
    """"""
    raise NotImplementedError