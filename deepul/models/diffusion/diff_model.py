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
    def __init__(self) -> None:
        super().__init__()
        
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