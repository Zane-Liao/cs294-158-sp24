import math
import random
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
    "UNet",
    "timestep_embedding",
    "ddpm_sample",
]


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        model,
        size=(),
        timesteps=None,
        objective='pred_v',
        offset_noise_strength=0.,  # https://www.crosslabs.org/blog/diffusion-with-offset-noise
        clip_range=None,
        ) -> None:
        super().__init__()
        self.model = model
        assert not (type(self) == GaussianDiffusion and self.model.in_dim != self.model.out_dim)
        assert not hasattr(self.model, 'random_or_learned_sinusoidal_cond') or not self.model.random_or_learned_sinusoidal_cond
        
        self.dim = getattr(self.model, 'in_dim', self.model.dim) or getattr(self.model, 'init_dim', self.model.dim)
        self.self_condition = self.model.self_condition
        
        self.size = size
        self.objective = objective
        self.offset_noise_strength = offset_noise_strength
        
        assert objective in {'pred_noise', 'pred_x0', 'pred_v'}, 'objective must be either pred_noise (predict noise) or pred_x0 (predict image start) or pred_v (predict v [v-parameterization as defined in appendix D of progressive distillation paper, used in imagen-video successfully])'

        self.num_timesteps = int(timesteps) if timesteps is not None else None
        
        if objective == 'pred_noise':
            self.snr_fn = lambda alpha, sigma: 1.
        elif objective == 'pred_x0':
            self.snr_fn = lambda alpha, sigma: alpha**2 / sigma**2
        elif objective == 'pred_v':
            self.snr_fn = lambda alpha, sigma: alpha**2
            
        if clip_range is not None:
            clip_min, clip_max = clip_range
            self.clip_fn = partial(torch.clamp, min=clip_min, max=clip_max)
        else:
            self.clip_fn = None
        
    @property
    def device(self):
        return self.model.device
    
    def _get_alpha_sigma(self, t):
        return torch.cos(t * math.pi / 2), torch.sin(t * math.pi / 2)
    
    def _expand(self, t):
        for _ in range(len(self.size)+1):
            t = t[..., None]
        return t
    
    def get_x(self, x_0, noise, alpha_t, sigma_t):
        return alpha_t * x_0 + sigma_t * noise
    
    def get_v(self, x_0, noise, alpha_t, sigma_t):
        return alpha_t * noise - sigma_t * x_0
    
    def predict_start_from_v(self, x_t, v, alpha_t, sigma_t):
        return alpha_t * x_t - sigma_t * v
    
    def predict_nosie_from_v(self, x_t, v, alpha_t, sigma_t):
        return sigma_t * x_t + alpha_t * v
    
    def predict_start_from_noise(self, x_t, noise, alpha_t, sigma_t):
        return (x_t - sigma_t * noise) / alpha_t
    
    def predict_noise_from_start(self, x_t, x_0, alpha_t, sigma_t):
        return (x_t - alpha_t * x_0) / sigma_t
    
    def forward(self, x, model_output, alpha_t, sigma_t, rederive_pred_noise=False):
        if self.objective == 'pred_noise':
            pred_noise = model_output

            x_0 = self.predict_start_from_noise(x, pred_noise, alpha_t, sigma_t)
            if self.clip_fn is not None:
                x_0 = self.clip_fn(x_0)
                if rederive_pred_noise:
                    pred_noise = self.predict_noise_from_start(x, x_0, alpha_t, sigma_t)

        elif self.objective == 'pred_x0':
            x_0 = model_output

            if self.clip_fn is not None:
                x_0 = self.clip_fn(x_0)
            pred_noise = self.predict_noise_from_start(x, x_0, alpha_t, sigma_t)

        elif self.objective == 'pred_v':
            v = model_output

            x_0 = self.predict_start_from_v(x, v, alpha_t, sigma_t)
            if self.clip_fn is not None:
                x_0 = self.clip_fn(x_0)
            pred_noise = self.predict_noise_from_start(x, x_0, alpha_t, sigma_t)

        return pred_noise, x_0
        
    @torch.inference_mode()
    def p_sample(self, x, t: int, x_self_cond=None):
        raise NotImplementedError
        
    @torch.inference_mode()
    def p_sample_loop(self, shape, return_all_timesteps=False):
        raise NotImplementedError
    
    # continuous time ddim
    @torch.inference_mode()
    def ddim_sample(self, n, labels=None, steps=None, eta=None, eps=1e-4, return_all_timesteps=False):
        if self.num_timesteps is not None:
            steps = default(steps, self.num_timesteps)
            assert steps <= self.num_timesteps
        
        # $t \sim \text{Uniform}(0, 1)$
        ts = torch.linspace(1 - eps, eps, steps=steps+1)
        
        # Init (n, dim, size)
        x = torch.randn(n, self.dim, *self.size, device=self.device)
        xs = [x]
        
        x_0 = None
        for i in range(steps):
            t_curr = torch.full((n,), ts[i], dtype=torch.float32, device=self.device)
            t_next = torch.full((n,), ts[i+1], dtype=torch.float32, device=self.device)
            
            # $\alpha_t = \cos\left(\frac{\pi}{2}t\right), \sigma_t = \sin\left(\frac{\pi}{2}t\right)$
            alpha_cur, sigma_cur = self._get_alpha_sigma(t_curr)
            alpha_next, sigma_next = self._get_alpha_sigma(t_next)
            
            # Broadcast Tensor
            alpha_cur, sigma_cur = self._expand(alpha_cur), self._expand(sigma_cur)
            alpha_next, sigma_next = self._expand(alpha_next), self._expand(sigma_next)
            
            self_cond = x_0 if self.self_condition else None
            model_output = self.model(x, t_curr, label=labels, x_self_cond=self_cond)
            
            # $\epsilon \sim N(0,I)$
            pred_noise, x_0 = self.forward(x, model_output, alpha_cur, sigma_cur, rederive_pred_noise=True)
            
            # $\eta_t = \sigma_{t-1}/\sigma_t\sqrt{1 - \alpha_t^2/\alpha_{t-1}^2}$
            eta_t = eta * (sigma_next / sigma_cur) * torch.sqrt(1 - alpha_cur**2 / alpha_next**2)
            
            noise = torch.randn_like(x)
            
            # $$x_{t-1} = \alpha_{t-1}\left(\frac{x_t - \sigma_t\hat{\epsilon}}{\alpha_t}\right) + \sqrt{\sigma_{t-1}^2 - \eta_t^2}\hat{\epsilon} + \eta_t\epsilon_t$$
            x = alpha_next * x_0 + torch.sqrt((sigma_next**2 - eta_t**2).clamp(min=0)) * pred_noise + eta_t * noise
            
            xs.append(x)
            
        return x if not return_all_timesteps else torch.stack(xs, dim=1)
        
    @torch.inference_mode()
    def sample(self, n=16, label=None, steps=512, eta=1., return_all_timesteps=False):
        if label is not None:
            if isinstance(label, int):
                labels = torch.LongTensor([label]*n)
            else:
                labels = label.repeat_interleave(n, dim=0)
            labels = labels.to(self.device)
            n_samples = len(labels)
        else:
            labels = None
            n_samples = n
            
        samples = self.ddim_sample(n_samples, labels=labels, steps=steps, eta=eta, return_all_timesteps=return_all_timesteps)
        if label is not None and not isinstance(label, int):
            samples = samples.reshape(-1, n, *samples.shape[1:])

        return samples.cpu().numpy()
        
    @torch.inference_mode()
    def interpolate(self, x1, x2, t=None, lam=0.5):
        raise NotImplementedError

    @torch.autocast('cuda', enabled = False)
    def q_sample(self, x_start, t, noise = None):
        raise NotImplementedError

    def p_losses(self, x_0, t, noise=None, y=None, offset_noise_strength=None):
        noise = default(noise, lambda: torch.randn_like(x_0))

        alpha_t, sigma_t = self._get_alpha_sigma(t)
        snr_t = self.snr_fn(alpha_t, sigma_t)

        alpha_t, sigma_t = self._expand(alpha_t), self._expand(sigma_t)

        # offset noise - https://www.crosslabs.org/blog/diffusion-with-offset-noise

        offset_noise_strength = default(offset_noise_strength, self.offset_noise_strength)

        if offset_noise_strength > 0.:
            offset_noise = torch.randn(x_0.shape[:2], device = self.device)
            while offset_noise.dim() < x_0.dim():
                offset_noise = offset_noise[..., None]
            noise += offset_noise_strength * offset_noise

        # noise sample
        x = self.get_x(x_0, noise, alpha_t, sigma_t)

        # if doing self-conditioning, 50% of the time, predict x0 from current set of times
        # and condition with unet with that
        # this technique will slow down training by 25%, but seems to lower FID significantly

        x_self_cond = None
        if self.self_condition and random() < 0.5:
            with torch.no_grad():
                model_output = self.model(x, t, label=y)

                if self.objective == 'pred_noise':
                    pred_noise = model_output

                    x_self_cond = self.predict_start_from_noise(x, pred_noise, alpha_t, sigma_t)
                    if self.clip_fn is not None:
                        x_self_cond = self.clip_fn(x_self_cond)
                elif self.objective == 'pred_x0':
                    x_self_cond = model_output

                    if self.clip_fn is not None:
                        x_self_cond = self.clip_fn(x_self_cond)
                elif self.objective == 'pred_v':
                    v = model_output

                    x_self_cond = self.predict_start_from_v(x, v, alpha_t, sigma_t)
                    if self.clip_fn is not None:
                        x_self_cond = self.clip_fn(x_self_cond)

                x_self_cond.detach_()

        # predict and take gradient step

        model_out = self.model(x, t, label=y, x_self_cond=x_self_cond)

        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_0
        elif self.objective == 'pred_v':
            target = self.get_v(x_0, noise, alpha_t, sigma_t)
        else:
            raise ValueError(f'unknown objective {self.objective}')

        loss = F.mse_loss(model_out, target, reduction='none')
        loss = torch.mean(loss, dim=tuple(range(1, loss.dim())))

        loss = loss * snr_t
        return loss.mean()

    def loss(self, x, y=None, *args, **kwargs):
        b, _, *size = x.shape
        for s, ss in zip(size, self.size):
            assert s == ss, f'size must be {self.size}'
        if self.num_timesteps is None:
            t = torch.rand((b,), device=x.device)
        else:
            t = torch.randint(0, self.num_timesteps, (b,), device=x.device) / self.num_timesteps

        return self.p_losses(x, t, y=y, *args, **kwargs)


class DiT(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
    

class DiTBlock(nn.Module):
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
        resnet_block = partial(ResnetBlock, time_emb_dim = time_dim, dropout = dropout)
        
        # layers
        
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)
        
        # DownSamples
        for ind, ((dim_in, dim_out),layer_full_attn, layer_attn_heads, layer_attn_dim_head) in enumerate(zip(in_out, full_attn, attn_heads, attn_dim_head)):
            is_last = ind >= (num_resolutions - 1)
            
            attn_klass = FullAttention if layer_full_attn else LinearAttention
            
            self.downs.append(nn.ModuleList([
                resnet_block(dim_in, dim_in),
                resnet_block(dim_in, dim_in),
                attn_klass(dim_in, heads=layer_attn_heads, dim_head=layer_attn_dim_head),
                DownSample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding=1),
            ]))
        
        # Mid
        mid_dim = dims[-1]
        self.mid_block1 = resnet_block(mid_dim, mid_dim)
        self.mid_attn = FullAttention(mid_dim, heads=attn_heads[-1], dim_head=attn_dim_head[-1])
        self.mid_block2 = resnet_block(mid_dim, dim_in)
        
        # UpSamples
        for ind, ((dim_in, dim_out), layer_full_attn, layer_attn_heads, layer_attn_dim_head) in enumerate(zip(*map(reversed, (in_out, full_attn, attn_heads, attn_dim_head)))):
            is_last = ind == (len(in_out) - 1)
            
            attn_klass = FullAttention if layer_full_attn else LinearAttention
            
            self.ups.append(nn.ModuleList([
                resnet_block(dim_out+dim_in, dim_out),
                resnet_block(dim_out+dim_in, dim_out),
                attn_klass(dim_out, heads=layer_attn_heads, dim_head=layer_attn_dim_head),
                UpSample(dim_out, dim_in) if not is_last else nn.Conv2d(dim_out, dim_in, 3, padding=1),
            ]))
            
        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)
        
        self.final_res_block = resnet_block(init_dim * 2, init_dim)
        self.final_conv = nn.Conv2d(init_dim, self.out_dim, 1)
    
    @property
    def downsample_factor(self):
        return 2 ** (len(self.downs) - 1)
        
    def forward(self, x: torch.Tensor, time, x_self_cond=None) -> torch.Tensor:
        assert all([divisible_by(d, self.downsample_factor) for d in x.shape[-2:]]), f'your input dimensions {x.shape[-2:]} need to be divisible by {self.downsample_factor}, given the unet'
        
        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim=1)
        
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