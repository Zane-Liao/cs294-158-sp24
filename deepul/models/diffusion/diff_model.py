import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from deepul.models.modules.layers import *
import numpy as np
from deepul.hw3_utils.lpips import default
from functools import partial
from timm.models.vision_transformer import Attention, Mlp

__all__ = [
    "GaussianDiffusion",
    "DiT",
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
        
        self.dim = self.model.in_dim
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
        return next(self.model.parameters()).device

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
    def __init__(
        self,
        input_shape,
        patch_size,
        hidden_size,
        num_heads,
        num_layers,
        num_classes,
        cfg_dropout_prob,
        ) -> None:
        super().__init__()
        self.input_shape = input_shape
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.cfg_dropout_prob = cfg_dropout_prob
        
        self.in_channels = input_shape[0]
        self.embedding = nn.Embedding(num_classes + 1, hidden_size)
        self.layers = nn.ModuleList([
            DiTBlock(hidden_size, num_heads)
            for _ in range(num_layers)
        ])
        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.final = FinalLayer(hidden_size, patch_size, input_shape[0])
        self.proj = nn.Linear(patch_size * patch_size * input_shape[0], hidden_size)
        
    def dropout_classes(self, y: torch.Tensor, cfg_dropout_prob):
        p = torch.rand(y.shape[0]) < cfg_dropout_prob
        y[p] = self.num_classes
        return y
        
    def forward(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x = patchify_flatten(x, self.patch_size)
        x = self.proj(x)
        grid_size = int(x.shape[1] ** 0.5)
        pos_embed = get_2d_sincos_pos_embed(self.hidden_size, grid_size)
        pos_embed = torch.from_numpy(pos_embed).float().to(x.device)
        x = x + pos_embed.unsqueeze(0)
        
        t = compute_timestep_embedding(t, self.hidden_size)
        t = self.time_mlp(t.to(x.dtype))
        if self.training:
            y = self.dropout_classes(y, self.cfg_dropout_prob)
        y = self.embedding(y)
        c = t + y
        
        for layer in self.layers: 
            x = layer(x, c)

        x = self.final(x, c)
        x = unpatchify(x, self.patch_size, self.input_shape[1], self.input_shape[2])
        
        return x

    @property
    def device(self):
        return next(self.parameters()).device
    
    def _get_alpha_sigma(self, t):
        return torch.cos(t * math.pi / 2), torch.sin(t * math.pi / 2)

    @torch.inference_mode()
    def sample(
        self,
        num_steps: int,
        batch_size: int,
        y: torch.Tensor,
        cfg_scale: float,
        cfg_rescale: float = 0.0,
        device: torch.device = None,
        seed: Optional[int] = None,
    ):
        self.eval()
        if device is None: device = self.device
        generator = torch.Generator(device=device)
        if seed is not None: generator.manual_seed(seed)

        ts = torch.linspace(1.0 - 1e-4, 1e-4, num_steps + 1, device=device)
        shape = (batch_size, self.in_channels, self.input_shape[1], self.input_shape[2])
        x = torch.randn(shape, device=device, generator=generator)
        y_null = torch.full_like(y, self.num_classes)
        
        alpha_all, sigma_all = self._get_alpha_sigma(ts)

        for i in range(num_steps):
            t_curr = ts[i]
            
            x_in = torch.cat([x, x], dim=0)
            t_in = torch.full((batch_size * 2,), float(t_curr), device=device)
            y_in = torch.cat([y, y_null], dim=0)
            
            noise_pred = self.forward(x_in, y_in, t_in)
            eps_cond, eps_uncond = noise_pred.chunk(2, dim=0)
            
            eps_hat = eps_uncond + cfg_scale * (eps_cond - eps_uncond)

            if cfg_rescale > 0.0:
                std_cond = eps_cond.std(dim=[1,2,3], keepdim=True)
                std_hat = eps_hat.std(dim=[1,2,3], keepdim=True)
                eps_hat_rescaled = eps_hat * (std_cond / (std_hat + 1e-8))
                eps_hat = eps_hat_rescaled * cfg_rescale + eps_hat * (1.0 - cfg_rescale)

            a_t = alpha_all[i].view(1, 1, 1, 1)
            s_t = sigma_all[i].view(1, 1, 1, 1)
            a_prev = alpha_all[i+1].view(1, 1, 1, 1)
            s_prev = sigma_all[i+1].view(1, 1, 1, 1)

            pred_x0 = (x - s_t * eps_hat) / a_t.clamp(min=1e-5)
            
            pred_x0 = pred_x0.clamp(-1.0, 1.0)

            dir_xt = s_prev * eps_hat
            x = a_prev * pred_x0 + dir_xt

        return x

    def loss(self, x: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        device = x.device
        b = x.shape[0]
        if y is None:
            y = torch.full((b,), self.num_classes, dtype=torch.long, device=device)
        
        t = torch.rand(b, device=device)
        eps = torch.randn_like(x, device=device)
        alpha, sigma = self._get_alpha_sigma(t)
        
        alpha = alpha.view(b, 1, 1, 1)
        sigma = sigma.view(b, 1, 1, 1)
        x_t = alpha * x + sigma * eps
        
        pred = self.forward(x_t, y, t)
        return F.mse_loss(pred, eps, reduction="mean")


class DiTBlock(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_heads,
        ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        
        self.silu = nn.SiLU()
        self.linear = nn.Linear(hidden_size, 6 * hidden_size)
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        mlp_hidden_ratio = int(hidden_size * 4.0)
        approx_silu = lambda: nn.SiLU()
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_ratio,
            act_layer=approx_silu,
        )
        
    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        c = self.silu(c)
        c = self.linear(c)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = c.chunk(6, dim=1)
        
        h = self.norm1(x)
        h = modulate(h, shift_msa, scale_msa)
        x = x + gate_msa.unsqueeze(1) * self.attn(h)
        
        h = self.norm2(x)
        h = modulate(h, shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(h)
        
        return x

def compute_timestep_embedding(timesteps: torch.Tensor, dim, max_period=10000):
    device = timesteps.device
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(0, half, dtype=torch.float32, device=device) / half)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
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

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb
    
def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb

def get_2d_sincos_pos_embed(embed_dim, grid_size):
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    return pos_embed

def patchify_flatten(x: torch.Tensor, patch_size):
    B, C, H, W = x.shape
    return x.view(B, C, H // patch_size, patch_size, W // patch_size, patch_size).permute(0, 2, 4, 1, 3, 5).reshape(B, -1, C * patch_size * patch_size)

def unpatchify(x: torch.Tensor, patch_size, H, W):
    B, L, D = x.shape
    C = D // (patch_size * patch_size)
    return x.reshape(B, H // patch_size, W // patch_size, C, patch_size, patch_size).permute(0, 3, 1, 4, 2, 5).reshape(B, C, H, W)