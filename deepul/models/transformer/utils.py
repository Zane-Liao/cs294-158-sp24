import math
from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Iterable, Optional, Tuple, Union

from jaxtyping import Float, Int

from deepul.models.attention.utils import MultiHeadAttention
from deepul.models.modules.layers import Linear, Embedding, LayerNorm, FFN

__all__ = [
    "GPT",
    "IGPT",
    "MMGPT",
]

class GPT(nn.Module):
    """"""
    def __init__(self,
                vocab_size: int,
                context_length: int,
                num_layers: int,
                d_model: int, 
                num_heads: int,
                d_ff: int,
                ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        
        self.context_length = context_length
        
        self.num_layers = num_layers
        
        self.embedding = Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
        
        self.position_embedding = Embedding()

        self.layers = nn.ModuleList(
            [
                GPTBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    max_seq_len=context_length,
                )
                for _ in range(num_layers)
            ]
        )

        self.ln_final = LayerNorm(d_model=d_model)
        self.lm_head = Linear(in_features=d_model, out_features=vocab_size)

    def forward(self, x: Int[torch.Tensor, "... sequence_length"]) -> Float[torch.Tensor, "... sequence_length vocab_size"]:
        _, sequence_length = x.size()

        x = self.embedding(x)
        
        for layer in self.layers:
            x = layer(x)
            
        x = self.ln_final(x)

        return self.lm_head(x)
    
    def loss(self) -> torch.Tensor:
        raise NotImplementedError
    
    def sample(self) -> torch.Tensor:
        raise NotImplementedError
    

class IGPT(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        raise NotImplementedError
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
    
    def loss(self) -> torch.Tensor:
        raise NotImplementedError
    
    def sample(self) -> torch.Tensor:
        raise NotImplementedError


class MMGPT(nn.Module):
    """Multimodal Transformer"""
    def __init__(self) -> None:
        super().__init__()
        raise NotImplementedError
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
    
    def loss(self) -> torch.Tensor:
        raise NotImplementedError
    
    def sample(self) -> torch.Tensor:
        raise NotImplementedError
    

class GPTBlock(nn.Module):
    """"""
    def __init__(self,
                d_model: int,
                num_heads: int,
                d_ff: int,
                max_seq_len: int | None = None,
                device=None,
                dtype=None,
                ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff

        self.layer_norm1 = LayerNorm(d_model, **factory_kwargs)

        self.self_attn = MultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            theta=None,
            max_seq_len=max_seq_len,
            rope_exist=None,
            **factory_kwargs,
            )

        self.layer_norm2 = LayerNorm(d_model, **factory_kwargs)
        self.ff = FFN(d_model, d_ff)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Create token_positions
        seq_len = x.shape[1]
        token_positions = torch.arange(seq_len, device=x.device)

        attn_output = self.self_attn(self.layer_norm1(x),
                                         token_positions=token_positions
                                        )
        y = x + attn_output
        
        return y + self.ff(self.layer_norm2(y))


def compute_lr(t, alpha_max, alpha_min, t_w, t_c) -> (Any | None):
    if t < t_w:
        return t / t_w * alpha_max
    if t_w <= t and t <= t_c:
        cosine = math.cos((t - t_w) / (t_c - t_w) * math.pi)
        return alpha_min + 0.5 * (1 + cosine) * (alpha_max - alpha_min)
    if t > t_c:
        return alpha_min

def gradient_cliping(parameters: Iterable[nn.Parameter], max_l2_norm: float, epsilon = 1e-6) -> None:
    total_norm_ = 0.0
    for p in parameters:
        if p.grad is None:
            continue
        total_norm_ += p.grad.data.norm(2).item() ** 2
    
    total_norm = math.sqrt(total_norm_)
    
    if total_norm > max_l2_norm:
        scale = max_l2_norm / (total_norm + epsilon)
        for p in parameters:
            if p.grad is None:
                continue
            p.grad.data.mul_(scale)