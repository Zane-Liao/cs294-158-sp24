import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Optional, Tuple, Union
from einops import einsum, rearrange, repeat

from packaging import version
from collections import namedtuple
from functools import partial
from torch.nn.attention import SDPBackend
from deepul.hw3_utils.lpips import exists, default, once, print_once, divisible_by

__all__ = [
    "MaskedConv2d",
    "MaskedResidualblock",
    "RotaryPositionalEmbedding",
    "Linear",
    "Embedding",
    "SwiGLU",
    "RMSNorm",
    "GELU",
    "FFN",
    "PositionalEncoding1D",
    "LayerNorm",
    "RotaryCausalSelfAttention",
    "CausalSelfAttention",
    "MultimodalCausalSelfAttention",
    "DepthToSpace",
    "SpaceToDepth",
    "Attend",
    "Attention",
    "LinearAttention",
    "MLP",
    "FeedForward",
    "TimeEmbedding",
    "UpSample",
    "DownSample",
    "Block",
    "Residualblock",
    "FinalLayer",
    "RMSNorm2d",
    "RMSNormConv",
    "SinusoidalPosEmb",
    "RandomOrLearnedSinusoidalPosEmb",
    "ResnetBlock",
    "TimeMLP",
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
                 ) -> None:
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
            if self.mask_type == 'B':
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
    

class MaskedResidualblock(nn.Module):
    """"""
    def __init__(self, channels: int, kernel_size: int = 7, _layer_norm: bool = True) -> None:
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
    

class Linear(nn.Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: torch.Tensor
    
    def __init__(self, in_features, out_features, bias=False, device=None, dtype=None) -> None:
        """
        linear transformation module.
    
        Parameters:
            in_features: int final dimension of the input
        
            out_features: int final dimension of the output
        
            device: torch.device | None = None Device to store the parameters on
        
            dtype: torch.dtype | None = None Data type of the parameters
        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight = nn.Parameter(
            torch.empty((out_features, in_features), **factory_kwargs)
        )
        
        self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs)) if bias else self.register_buffer("bias", None)
        
        std = math.sqrt(2 / (in_features + out_features))
        
        nn.init.trunc_normal_(
            self.weight, mean=0.0, std=std, a=-3.0*std, b=3.0*std
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the linear transformation to the input.
        
        Parameter:
            x: torch.Tensor
        Return:
            torch.Tensor
        """
        output = x @ self.weight.T
        if self.bias is not None:
            output = output + self.bias
        return output
    

class Embedding(nn.Module):
    __constants__ = ['num_embeddings', 'embedding_dim']
    num_embeddings: int
    embedding_dim: int
    weight: torch.Tensor
    
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None) -> None:
        """
        an embedding module.
        
        Parameters:
            num_embeddings: int Size of the vocabulary
            
            embedding_dim: int Dimension of the embedding vectors, i.e., dmodel
            
            device: torch.device | None = None Device to store the parameters on
            
            dtype: torch.dtype | None = None Data type of the parameters
        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(
            torch.empty((num_embeddings, embedding_dim), **factory_kwargs)
        )
        nn.init.trunc_normal_(
            self.weight, mean=0.0, std=1.0, a=-3.0, b=3.0
        )
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Lookup the embedding vectors for the given token IDs.
        
        Parameter:
            token_ids: torch.Tensor
        Return:
            torch.Tensor
        """
        return torch.embedding(self.weight, token_ids)


class RotaryPositionalEmbedding(nn.Module):
    """"""
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None) -> None:
        super().__init__()
        half_dim = d_k // 2
        self.theta = (1. / theta ** (torch.arange(0, half_dim, device=device) / half_dim))
        
        self.max_seq_len = torch.arange(max_seq_len, device=device)
        angle_max_seq_len = torch.einsum("i,j->ij", self.max_seq_len, self.theta)

        self.register_buffer("sin", torch.sin(angle_max_seq_len), persistent=False)
        self.register_buffer("cos", torch.cos(angle_max_seq_len), persistent=False)
    
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
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

        self.w1 = Linear(d_model, d_ff) 
        self.w2 = Linear(d_ff, d_model)
        self.w3 = Linear(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(silu(self.w1(x)) * self.w3(x))


def silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


class RMSNorm(nn.Module):
    """"""
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, **factory_kwargs))
        
    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # In torch, x.float() => float32
        output = self._norm(x.float()).type_as(x)
        return output * self.weight
    

class GELU(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return gelu(x)


# Impl: https://github.com/pytorch/pytorch/issues/20464
# $\text{GELU}(x) = 0.5 * x * (1 + \text{Tanh}(\sqrt{2 / \pi} * (x + 0.044715 * x^3)))$
def gelu(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * x * (1 + torch.tanh(torch.sqrt(torch.tensor(2. / torch.pi, dtype=x.dtype, device=x.device)) * (x + 0.044715 * x ** 3)))

class FFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int, prob: float = 0.01) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        
        self.w1 = Linear(d_model, d_ff)
        self.gelu = GELU()
        self.w2 = Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(self.gelu(self.w1(x))))


# Impl: https://github.com/hyunwoongko/transformer
class PositionalEncoding1D(nn.Module):
    def __init__(self, d_model: int, max_len: int, device=None) -> None:
        super().__init__()
        
        self.encoding = torch.zeros(max_len, d_model, device=device)
        self.encoding.requires_grad = False # Not Need Compute Gradient
        
        pos = torch.arange(0, max_len, device=device)
        pos = pos.float().unsqueeze(-1)
        
        _2i = torch.arange(0, d_model, step=2, device=device).float()
        
        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, seq_len = x.size()
        
        return self.encoding[:seq_len, :]


# Impl: https://github.com/hyunwoongko/transformer
# $y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta$
class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6, device=None, dtype=None) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        
        self.gamma = nn.Parameter(torch.ones(d_model, **factory_kwargs))
        self.beta = nn.Parameter(torch.zeros(d_model, **factory_kwargs))
        self.eps = eps
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(-1, keepdim=True)
        
        var = x.var(-1, unbiased=False, keepdim=True)
        
        x_bar = (x - mean) / torch.sqrt(var + self.eps)
        
        return self.gamma * x_bar + self.beta
    

class RotaryCausalSelfAttention(nn.Module):
    """"""
    def __init__(self,
                 d_model: int,
                 num_heads: int,
                 theta: float | None = None,
                 max_seq_len: int | None = None,
                 rope_exist: bool | None = None,
                 device=None,
                 dtype=None,
            ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.qkv_proj = Linear(d_model, 3 * d_model, **factory_kwargs)
        self.o_proj = Linear(d_model, d_model, **factory_kwargs)
        
        self.rope_exist = rope_exist
        if self.rope_exist:
            self.rope = RotaryPositionalEmbedding(
                theta=theta,
                d_k=self.d_k,
                max_seq_len=max_seq_len,
                device=device
            )
        else:
            self.rope = None

    def forward(self,
                in_features: torch.Tensor,
                token_positions: Optional[torch.Tensor] = None,
               ) -> torch.Tensor:
        
        batch_size, seq_len, _ = in_features.shape

        qkv = self.qkv_proj(in_features).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b t (h d) -> b h t d", h=self.num_heads), qkv)

        if self.rope_exist:
            if token_positions is None:
                raise ValueError("token_positions must be provided when use_rope is True.")
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)
        
        output = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        
        return self.o_proj(rearrange(output, "b h t d -> b t (h d)"))
    
    
class CausalSelfAttention(nn.Module):
    """"""
    def __init__(self,
                 d_model: int,
                 num_heads: int,
                 device=None,
                 dtype=None,
                 ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        
        self.d_k = d_model // num_heads
        
        self.qkv_proj = nn.Linear(d_model, 3*d_model, **factory_kwargs)
        self.o_proj = nn.Linear(d_model, d_model, **factory_kwargs)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qkv = self.qkv_proj(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b t (h d) -> b h t d", h=self.num_heads), qkv)
        
        output = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        
        return self.o_proj(rearrange(output, "b h t d -> b t (h d)"))


class MultimodalCausalSelfAttention(nn.Module):
    """1D/2D/3D Space AR"""
    def __init__(
        self,
        in_channels: int,
        embed_channels: int = None,
        out_channels: int = None,
        num_heads: int = 1,
        n_dims: int = 2,
        mask_current: bool = True,
        extra_input_channels: int = 0,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        
        self.n_dims = n_dims
        self._num_heads = num_heads
        self._mask_current = mask_current
        
        self._embed_channels = embed_channels or in_channels
        self._out_channels = out_channels or in_channels

        assert self._embed_channels % num_heads == 0, \
            f"embed_channels ({self._embed_channels}) must be divisible by num_heads ({num_heads})"
        assert self._out_channels % num_heads == 0, \
            f"out_channels ({self._out_channels}) must be divisible by num_heads ({num_heads})"
        
        self.head_dim_q = self._embed_channels // num_heads
        self.head_dim_v = self._out_channels // num_heads

        mod = nn.Conv1d if n_dims == 1 else nn.Conv2d if n_dims == 2 else nn.Conv3d
        
        self._q = mod(in_channels, self._embed_channels, 1, 1, 0, **factory_kwargs)

        total_in_channels = in_channels + extra_input_channels
        total_out_channels = self._embed_channels + self._out_channels
        self._kv = mod(total_in_channels, total_out_channels, 1, 1, 0, **factory_kwargs)
        
        # Output Projection
        self._o_proj = mod(self._out_channels, self._out_channels, 1, 1, 0, **factory_kwargs)

    def _flatten_spatial(self, t):
        return rearrange(t, 'n c ... -> n c (...)')

    def forward(self, x: torch.Tensor, extra_x: torch.Tensor = None, pos: torch.Tensor = None) -> torch.Tensor:
        b, c, *spatial_shape = x.shape
        L_query = np.prod(spatial_shape)

        q_input = x 
        if pos is not None:
            q_input = q_input + pos
            
        q = self._q(q_input)
        q = self._flatten_spatial(q)
        q = rearrange(q, 'b (h d) l -> b h l d', h=self._num_heads)

        if extra_x is not None:
            kv_input = torch.cat([x, extra_x], dim=1)
        else:
            kv_input = x
            
        kv = self._kv(kv_input)
        k_raw, v_raw = torch.split(kv, [self._embed_channels, self._out_channels], dim=1)
        
        if pos is not None:
            k_raw = k_raw + pos 

        k = self._flatten_spatial(k_raw)
        v = self._flatten_spatial(v_raw)
        
        k = rearrange(k, 'b (h d) s -> b h s d', h=self._num_heads)
        v = rearrange(v, 'b (h d) s -> b h s d', h=self._num_heads)

        S_key = L_query 
        diag_val = 0 if self._mask_current else -1
        attn_mask = torch.ones((L_query, S_key), device=x.device, dtype=torch.bool)
        attn_mask = torch.tril(attn_mask, diagonal=diag_val)
        
        out = F.scaled_dot_product_attention(
            q, k, v, 
            attn_mask=attn_mask, 
            dropout_p=0.0, 
            is_causal=False
        )
        out = rearrange(out, 'b h l d -> b (h d) l')

        spatial_dict = {f'd{i}': s for i, s in enumerate(spatial_shape)}
        if self.n_dims == 1:
            out = rearrange(out, 'b c (d0) -> b c d0', **spatial_dict)
        elif self.n_dims == 2:
            out = rearrange(out, 'b c (d0 d1) -> b c d0 d1', **spatial_dict)
        elif self.n_dims == 3:
            out = rearrange(out, 'b c (d0 d1 d2) -> b c d0 d1 d2', **spatial_dict)
            
        return self._o_proj(out)
    

class DepthToSpace(nn.Module):
    def __init__(self, block_size):
        super().__init__()
        self.block_size = block_size
        self.block_size_sq = block_size * block_size

    def forward(self, input):
        output = input.permute(0, 2, 3, 1)
        (batch_size, d_height, d_width, d_depth) = output.size()
        s_depth = int(d_depth / self.block_size_sq)
        s_width = int(d_width * self.block_size)
        s_height = int(d_height * self.block_size)
        t_1 = output.reshape(batch_size, d_height, d_width, self.block_size_sq, s_depth)
        spl = t_1.split(self.block_size, 3)
        stack = [t_t.reshape(batch_size, d_height, s_width, s_depth) for t_t in spl]
        output = torch.stack(stack, 0).transpose(0, 1).permute(0, 2, 1, 3, 4).reshape(batch_size, s_height, s_width,
                                                                                      s_depth)
        output = output.permute(0, 3, 1, 2)
        return output
    
    
class SpaceToDepth(nn.Module):
    def __init__(self, block_size):
        super().__init__()
        self.block_size = block_size
        self.block_size_sq = block_size * block_size

    def forward(self, input):
        output = input.permute(0, 2, 3, 1)
        (batch_size, s_height, s_width, s_depth) = output.size()
        d_depth = s_depth * self.block_size_sq
        d_width = int(s_width / self.block_size)
        d_height = int(s_height / self.block_size)
        t_1 = output.split(self.block_size, 2)
        stack = [t_t.reshape(batch_size, d_height, d_depth) for t_t in t_1]
        output = torch.stack(stack, 1)
        output = output.permute(0, 2, 1, 3)
        output = output.permute(0, 3, 1, 2)
        return output


AttentionConfig = namedtuple('AttentionConfig', ['backends'])

# credits: https://github.com/lucidrains/denoising-diffusion-pytorch/blob/main/denoising_diffusion_pytorch/attend.py
class Attend(nn.Module):
    """"""
    def __init__(
        self,
        dropout = 0.,
        flash = False,
        scale = None,
        ) -> None:
        super().__init__()
        self.dropout = dropout
        self.scale = scale
        self.attn_dropout = nn.Dropout(dropout)
        
        self.flash = flash
        assert not (flash and version.parse(torch.__version__)) < version.parse('2.0.0'), "in order to use flash attention, you must be using pytorch 2.0 or above"
        
        # determine efficient attention configs for cuda and cpu
        
        self.cpu_config = AttentionConfig([SDPBackend.FLASH_ATTENTION, SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION])
        self.cuda_config = None
        
        if not torch.cuda.is_available() or not flash:
            return
        
        device_properties = torch.cuda.get_device_properties(torch.device('cuda'))

        device_version = version.parse(f'{device_properties.major}.{device_properties.minor}')
        
        if device_version > version.parse('8.0'):
            print_once('A100 GPU detected, using flash attention if input tensor is on cuda')
            self.cuda_config = AttentionConfig([SDPBackend.FLASH_ATTENTION])
        else:
            print_once('Non-A100 GPU detected, using math or mem efficient attention if input tensor is on cuda')
            self.cuda_config = AttentionConfig([SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION])
            
    def flash_attn(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        _, heads, q_len, _, k_len, is_cuda, device = *q.shape, k.shape[-2], q.is_cuda, q.device
        
        if exists(self.scale):
            default_scale = q.shape[-1]
            q = q * (self.scale / default_scale)
        
        q, k, v = map(lambda t: t.contiguous(), (q, k, v))
        
        # Check if there is a compatible device for flash attention

        config = self.cuda_config if is_cuda else self.cpu_config

        # pytorch 2.0 flash attn: q, k, v, mask, dropout, causal, softmax_scale
        
        with torch.nn.attention.sdpa_kernel(**config._asdict()):
            out = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.dropout if self.training else 0
            )
        
        return out

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        einstein notation
        b - batch
        h - heads
        n, i, j - sequence length (base sequence length, source, target)
        d - feature dimension
        """
        q_len, k_len, device = q.shape[-2], k.shape[-2], q.device
        
        if self.flash:
            return self.flash_attn(q, k, v)

        scale = default(self.scale, q.shape[-1] ** 0.5)
        
        # similarity
        
        sim = einsum(q, k, "b h i d, b h j d -> b h i j") * scale
        
        # attention
        
        attn = sim.softmax(dim=-1)
        attn = self.attn_dropout(attn)
        
        return einsum(attn, v, "b h i j, b h j d -> b h i d")


class Attention(nn.Module):
    """"""
    def __init__(
        self,
        dim, 
        heads = 4,
        dim_head = 32,
        num_mem_kv = 4,
        flash = False,
        ) -> None:
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        
        self.norm = RMSNormConv(dim)
        self.attend = Attend(flash=flash)
        
        self.mem_kv = nn.Parameter(torch.randn(2, heads, num_mem_kv, dim_head))
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
    
        x = self.norm(x)
        
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, "b (h c) x y -> b h (x y) c", h=self.heads), qkv)
        
        mk, mv = map(lambda t: repeat(t, "h n d -> b h n d", b=b), self.mem_kv)
        k, v = map(partial(torch.cat, dim=-2), ((mk, k), (mv, v)))
        
        out = self.attend(q, k, v)
        
        return self.to_out(rearrange(out, "b h (x y) d -> b (h d) x y", x=h, y=w))


class LinearAttention(nn.Module):
    """"""
    def __init__(
        self,
        dim, 
        heads = 4,
        dim_head = 32,
        num_mem_kv = 4,
        ) -> None:
        super().__init__()
        self.scale = dim_head * -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        
        self.norm = RMSNormConv(dim)
        self.mem_kv = nn.Parameter(torch.randn(2, heads, num_mem_kv, dim_head))
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        
        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),
            RMSNormConv(dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        
        x = self.norm(x)
        
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv)
        
        mk, mv = map(lambda t: repeat(t, "h c n -> b h c n", b=b), self.mem_kv)
        k, v = map(partial(torch.cat, dim=-1), ((mk, k), (mv, v)))
        
        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)
        
        q = q * self.scale
        
        context = einsum(k, v, "b h d n, b h e n -> b h d e")
        out = einsum(context, q, "b h d e, b h d n -> b h e n")
        
        return self.to_out(rearrange(out, "b h c (x y) -> b (h c) x y", h=self.heads, x=h, y=w))
    
    
class MLP(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, out_dim),
        )
        
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x, t], dim=-1))
    

class FeedForward(nn.Module):
    """"""
    def __init__(self) -> None:
        super().__init__()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    
class TimeEmbedding(nn.Module):
    def __init__(self, n_out: int, t_emb_dim: 128) -> None:
        super().__init__()
        
        self.te_block = nn.Sequential(nn.SiLU, nn.Linear(t_emb_dim, n_out))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.te_block(x)


def UpSample(dim, dim_out=None):
    return nn.Sequential(
        nn.Upsample(scale_factor = 2, mode = 'nearest'),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding = 1)
    )


def DownSample(dim, dim_out=None):
    return nn.Sequential(
        rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w', p1 = 2, p2 = 2),
        nn.Conv2d(dim * 4, default(dim_out, dim), 1)
    )


class Block(nn.Module):
    def __init__(self, dim, dim_out, dropout = 0.):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding = 1)
        self.norm = RMSNormConv(dim_out)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, scale_shift = None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return self.dropout(x)


class Residualblock(nn.Module):
    """Given x, temb"""
    def __init__(self) -> None:
        super().__init__()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x
    

class RMSNorm2d(nn.Module):
    """"""
    def __init__(self, dim, eps=1e-8) -> None:
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.eps = eps
    
    @torch.compile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.mean(x ** 2, dim=1, keepdim=True)
        return x * torch.rsqrt(rms + self.eps) * self.g
    
    
class RMSNormConv(nn.Module):
    def __init__(self, dim, n_dims=2):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, *[1]*n_dims))

    def forward(self, x):
        return F.normalize(x, dim=1) * self.g * (x.shape[1] ** 0.5)
    
    
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim, theta = 10000) -> None:
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(self.theta) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb
    
    
class RandomOrLearnedSinusoidalPosEmb(nn.Module):
    """ following @crowsonkb 's lead with random (learned optional) sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim, is_random = False) -> None:
        super().__init__()
        assert divisible_by(dim, 2)
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad = not is_random)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        fouriered = torch.cat((x, fouriered), dim = -1)
        return fouriered


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim = None, dropout = 0.):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, dropout = dropout)
        self.block2 = Block(dim_out, dim_out)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb = None):

        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            scale_shift = time_emb.chunk(2, dim = 1)

        h = self.block1(x, scale_shift = scale_shift)

        h = self.block2(h)

        return h + self.res_conv(x)
    
    
class TimeMLP(nn.Module):
    def __init__(
        self,
        dim,
        in_dim=3,
        out_dim=None,
        emb_dim=None,
        num_layers=4,
        num_classes=1,
        self_condition=False,
        learned_variance=False,
        learned_sinusoidal_cond=False,
        random_fourier_features=False,
        learned_sinusoidal_dim=16,
        sinusoidal_pos_emb_scale=1000,
        sinusoidal_pos_emb_theta=10000
    ):
        super().__init__()

        # determine dimensions

        self.in_dim = in_dim
        self.self_condition = self_condition
        input_dim = self.in_dim * (2 if self_condition else 1)

        emb_dim = default(emb_dim, dim)
        self.init_emb = nn.Linear(input_dim, emb_dim)

        self.out_dim = default(out_dim, in_dim)
        output_dim = self.out_dim * (1 if not learned_variance else 2)

        # time embeddings

        time_dim = emb_dim

        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(emb_dim, scale=sinusoidal_pos_emb_scale, theta=sinusoidal_pos_emb_theta)
            fourier_dim = emb_dim

        self.time_emb = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # class embeddings

        if num_classes > 1:
            self.label_emb = nn.Embedding(num_classes + 1, time_dim, padding_idx=0)
        else:
            self.register_parameter('label_emb', None)

        # layers

        dims = [2*emb_dim, *[dim]*num_layers, output_dim]
        in_out = list(zip(dims[:-1], dims[1:]))

        layers = []
        for ind, (dim_in, dim_out) in enumerate(in_out):
            layers.extend([
                nn.ReLU(),
                nn.Linear(dim_in, dim_out)
            ])
        self.layers = nn.Sequential(*layers)

    @property
    def device(self):
        return self.init_emb.weight.device

    def forward(self, x, time, label=None, x_self_cond=None):
        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim=1)

        x = self.init_emb(x)
        t = self.time_emb(time)
        if self.label_emb is not None:
            label = torch.zeros_like(time, dtype=torch.long) if label is None else label + 1
            y = self.label_emb(label)
            t = t + y

        return self.layers(torch.cat([x, t], dim=1))
