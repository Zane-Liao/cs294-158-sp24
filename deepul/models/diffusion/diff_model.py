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
from deepul.hw3_utils.lpips import exists, default, once, print_once, cast_tuple, divisible_by
from functools import partial

__all__ = [
    "GaussianDiffusion",
    "DiT",
    "timestep_embedding",
    "ddpm_sample",
]


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        
        ) -> None:
        super().__init__()
    
    @property
    def device(self):
        raise NotImplementedError
    
    def predict_start_from_noise(self, x_t, t, noise):
        raise NotImplementedError
    
    def predict_noise_from_start(self, x_t, t, x0):
        raise NotImplementedError
        
    def predict_v(self, x_start, t, noise):
        raise NotImplementedError
        
    def predict_start_from_v(self, x_t, t, v):
        raise NotImplementedError
        
    def q_posterior(self, x_start, x_t, t):
        raise NotImplementedError
        
    def model_predictions(self, x, t, x_self_cond = None, clip_x_start = False, rederive_pred_noise = False):
        raise NotImplementedError
        
    def p_mean_variance(self, x, t, x_self_cond = None, clip_denoised = True):
        raise NotImplementedError
        
    @torch.inference_mode()
    def p_sample(self, x, t: int, x_self_cond = None):
        raise NotImplementedError
        
    @torch.inference_mode()
    def p_sample_loop(self, shape, return_all_timesteps = False):
        raise NotImplementedError
        
    @torch.inference_mode()
    def ddim_sample(self, shape, return_all_timesteps = False):
        raise NotImplementedError
        
    @torch.inference_mode()
    def sample(self, batch_size = 16, return_all_timesteps = False):
        raise NotImplementedError
        
    @torch.inference_mode()
    def interpolate(self, x1, x2, t = None, lam = 0.5):
        raise NotImplementedError
        
    def noise_assignment(self, x_start, noise):
        raise NotImplementedError

    @torch.autocast('cuda', enabled = False)
    def q_sample(self, x_start, t, noise = None):
        raise NotImplementedError

    def p_losses(self, x_start, t, noise = None, offset_noise_strength = None):
        raise NotImplementedError

    def loss(self):
        raise NotImplementedError
        
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
        
        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        
        time_dim = dim * 4
        
        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features
        
        # Time Embedding---positional encoding
        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim, theta=sinusoidal_pos_emb_theta)
            fourier_dim = dim
        
        # TimeMLP
        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )
        
        # attention
        
        if not full_attn:
            full_attn = (*((False,) * (len(dim_mults) - 1)), True)
            
        num_stages = len(dim_mults)
        full_attn = cast_tuple(full_attn, num_stages)
        attn_heads = cast_tuple(attn_heads, num_stages)
        attn_dim_head = cast_tuple(attn_dim_head, num_stages)
        
        assert len(full_attn) == len(dim_mults)
        
        # prepare blocks
        
        FullAttention = partial(Attention, flash = flash_attn)
        resnet_block = partial(ResnetBlock, time_emb_im = time_dim, dropout = dropout)
        
        # layers
        
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)
        
        # DownSamples
        for ind, ((dim_in, dim_out),layer_full_attn, layer_attn_heads, layer_attn_dim_head) in enumerate(zip(in_out, full_attn, attn_heads, attn_dim_head)):
            is_last >= (num_resolutions - 1)
            
            attn_klass = FullAttention if layer_full_attn else LinearAttention
            
            self.downs.append(nn.ModuleList([
                resnet_block(dim_in, dim_in),
                resnet_block(dim_in, dim_in),
                attn_klass(dim_in, heads=layer_attn_heads, dim_head=layer_attn_dim_head),
                DownSample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding=1),
            ]))
        
        # Mid
        mid_dim = dim[-1]
        self.mid_block1 = resnet_block(mid_dim, mid_dim)
        self.mid_attn = FullAttention(mid_dim, heads=attn_heads[-1], dim_head=attn_dim_head[-1])
        self.mid_block2 = resnet_block(mid_dim, dim_in)
        
        # UpSamples
        for ind, ((dim_in, dim_out), layer_full_attn, layer_attn_heads, layer_attn_dim_head) in enumerate(zip(*map(reversed, (in_out, full_attn, attn_heads, attn_dim_head)))):
            is_last = ind == len(in_out) - 1
            
            attn_klass = FullAttention if layer_full_attn else LinearAttention
            
            self.ups.append(nn.ModuleList([
                resnet_block(dim_out+dim_in, dim_out),
                resnet_block(dim_out+dim_in, dim_out),
                attn_klass(dim_out, heads=layer_attn_heads, dim_head=layer_attn_dim_head),
                UpSample(dim_out, dim_in) if not is_last else nn.Conv2d(dim_out, dim_in, 3, padding=1),
            ]))
            
        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim,default_out_dim)
        
        self.final_res_block = resnet_block(init_dim * 2, init_dim)
        self.final_conv = nn.Conv2d(init_dim, self.out_dim, 1)
    
    @property
    def downsample_factor(self):
        return 2 ** (len(self.downs) - 1)
        
    def forward(self, x: torch.Tensor, time, x_self_cond = None) -> torch.Tensor:
        assert all([divisible_by(d, self.downsample_factor) for d in x.shape[-2:]]), f'your input dimensions {x.shape[-2:]} need to be divisible by {self.downsample_factor}, given the unet'
        
        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim = 1)
        
        x = self.init_conv(x)
        r = x.clone()
        
        t = self.time_mlp(time)
        
        h = []
        
        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)
            
            x = block2(x, t)
            x = attn(x) + x
            h.append(x)
            
            x = downsample(x)
            
        x = self.mid_block1(x, t)
        x = self.mid_attn(x) + x
        x = self.mid_block2(x, t)
        
        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)
            
            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, t)
            x = upsample(x)
            
        x = torch.cat((x, r), dim=1)
        
        x = self.final_res_block(x, t)
        
        return self.final_conv(x)

def timestep_embedding(timesteps, dim, max_period=10000):
    half = dim // 2
    freqs = np.exp(-np.log(max_period) * np.arange(0, half, dtype=torch.float32) / half)
    args = timesteps[:, None].astype(torch.float32) * freqs[None]
    embedding = torch.cat([np.cos(args), np.sin(args)], axis=-1)
    if dim % 2:
        embedding = torch.cat([embedding, np.zeros_like(embedding[:, :1])], axis=-1)
    return embedding

# gaussian diffusion trainer class

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def linear_beta_schedule(timesteps):
    """
    linear schedule, proposed in original ddpm paper
    """
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

def sigmoid_beta_schedule(timesteps, start = -3, end = 3, tau = 1, clamp_min = 1e-5):
    """
    sigmoid schedule
    proposed in https://arxiv.org/abs/2212.11972 - Figure 8
    better for images > 64x64, when used during training
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

def ddpm_sample():
    """"""
    raise NotImplementedError