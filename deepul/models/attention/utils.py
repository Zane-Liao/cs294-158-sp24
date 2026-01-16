from einops import einsum, rearrange
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union

from deepul.models.modules.layers import Linear, RotaryPositionalEmbedding

__all__ = [
    "RotaryCausalSelfAttention",
    "CausalSelfAttention",
    "MultimodalCausalSelfAttention",
    "MultiHeadAttention",
    "LinearAttention",
]

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

        qkv = self.qkv_proj(in_features)
        q, k, v = qkv.chunk(3, dim=-1)
        
        q = rearrange(q, "b t (h d) -> b h t d", h=self.num_heads)
        k = rearrange(k, "b t (h d) -> b h t d", h=self.num_heads)
        v = rearrange(v, "b t (h d) -> b h t d", h=self.num_heads)

        if self.rope_exist:
            if token_positions is None:
                raise ValueError("token_positions must be provided when use_rope is True.")
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)
        
        output = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        
        return self.o_proj(rearrange(output, "b h t d -> b t (h d)"))
    
    
class CausalSelfAttention(nn.Module):
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
        
        self.qkv_proj = Linear(d_model, 3*d_model, **factory_kwargs)
        self.o_proj = Linear(d_model, d_model, **factory_kwargs)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)
        
        q = rearrange(q, "b t (h d) -> b h t d", h=self.num_heads)
        k = rearrange(k, "b t (h d) -> b h t d", h=self.num_heads)
        v = rearrange(v, "b t (h d) -> b h t d", h=self.num_heads)
        
        output = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        
        return self.o_proj(rearrange(output, "b h t d -> b t (h d)"))
    
    
class MultimodalCausalSelfAttention(nn.Module):
    """1D/2D/3D Space AR, Not Sequence"""
    def __init__(
        self,
        in_channels,
        embed_channels=None,
        out_channels=None,
        num_heads=1,
        n_dims=2,
        mask_current=False,
        extra_input_channels=0,
        device=None,
        dtype=None,
        ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self._num_heads = num_heads
        self._embed_channels = embed_channels or in_channels
        self._out_channels = out_channels or in_channels
        self._mask_current = mask_current
        
        mod = nn.Conv1d if n_dims == 1 else nn.Conv2d if n_dims == 2 else nn.Conv3d
        
        self._q = mod(in_channels, self._embed_channels, 1, 1, 0, **factory_kwargs)
        self._kv = mod(in_channels+extra_input_channels, self._embed_channels + self._out_channels, 1, 1, 0, **factory_kwargs)
        self._o_proj = mod(self._out_channels, self._out_channels, 1, 1, 0, **factory_kwargs)
    
    def _to_multihead(self, t):
        return rearrange(t, 'n (h d) ... -> n h (...) d', h=self._num_heads)
    
    def forward(self, x: torch.Tensor, extra_x=None, pos=None) -> torch.Tensor:
        n, _, *shape = x.shape
        
        causal_mask = torch.triu(
            torch.ones(np.prod(shape), self._mask_current, x.device, x.dtype),
            diagonal=-int(self._mask_current)
        )
        
        mask = rearrange(causal_mask, '... -> 1 1 (...) (...)').to(next(self.parameters()).device)
        
        q = self._to_multihead(self._q(x))
        if extra_x is not None:
            x = rearrange(torch.stack([x, extra_x], dim=0), 'b n c h w -> n (b c) h w')
        kv = self._kv(x)
        k, v = rearrange(kv, 'b (ke ve) ... -> b ke ... ve', ke=self._embed_channels, ve=self._out_channels)
        k, v = self._to_multihead(k), self._to_multihead(v)
        
        attn = einsum('n h t d, n h s d -> n h t s', q, k) / np.sqrt(k.shape[-1])
        attn = attn.masked_fill(mask == 0, -np.inf)
        attn = F.softmax(attn, dim=-1).masked_fill(mask == 0, 0)
        
        out = einsum('n h t s, n h s d -> n h t d', attn, v)
        out = rearrange(out, 'n h t d -> n (h d) t')
        out = rearrange(out, 'n c (...) -> n c ...')
        
        return self._o_proj(out)

    
class LinearAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        raise NotImplementedError
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
    
    
class MultiHeadAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        raise NotImplementedError
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError