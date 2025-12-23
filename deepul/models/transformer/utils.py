import math
from einops import einsum, rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Iterable, Optional, Tuple, Union

from jaxtyping import Float, Int

from deepul.models.attention.utils import RotaryCausalSelfAttention, CausalSelfAttention
from deepul.models.modules.layers import Linear, Embedding, LayerNorm, FFN, RMSNorm, SwiGLU, PositionalEncoding1D
import numpy as np
import numpy.typing as npt

__all__ = [
    "GPT",
    "IGPT",
    "MMGPT",
    "compute_lr",
    "gradient_cliping",
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
                rope_theta: int,
                ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        
        self.context_length = context_length
        
        self.num_layers = num_layers
        
        self.embedding = Embedding(num_embeddings=vocab_size, embedding_dim=d_model)

        self.layers = nn.ModuleList(
            [
                GPTBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    theta=rope_theta,
                    max_seq_len=context_length,
                )
                for _ in range(num_layers)
            ]
        )

        self.ln_final = RMSNorm(d_model=d_model)
        self.lm_head = Linear(in_features=d_model, out_features=vocab_size)

    def forward(self, x: Int[torch.Tensor, "... sequence_length"]) -> Float[torch.Tensor, "... sequence_length vocab_size"]:
        _, sequence_length = x.size()

        x = self.embedding(x)
        
        for layer in self.layers:
            x = layer(x)
            
        x = self.ln_final(x)

        return self.lm_head(x)
    
    def loss(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
    
    def sample(self) -> torch.Tensor:
        raise NotImplementedError
    

class IGPT(nn.Module):
    """"""
    def __init__(self,
                vocab_size: int,
                num_layers: int,
                d_model: int, 
                num_heads: int,
                d_ff: int,
                max_seq_len: int,
                device=None,
                dtype=None,
                ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.vocab_size = vocab_size
        
        self.num_layers = num_layers
        
        self.d_model = d_model
        
        self.embedding = Embedding(num_embeddings=vocab_size, embedding_dim=d_model, **factory_kwargs)

        self.layers = nn.ModuleList(
            [
                IGPTBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    **factory_kwargs,
                )
                for _ in range(num_layers)
            ]
        )

        self.ln_final = LayerNorm(d_model=d_model, **factory_kwargs)
        self.lm_head = Linear(in_features=d_model, out_features=vocab_size, **factory_kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, sequence_length = x.size()

        x = self.embedding(x)
        
        for layer in self.layers:
            x = layer(x)
            
        x = self.ln_final(x)

        return self.lm_head(x)
    
    def loss(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(next(self.parameters()).device)
        
        inputs = x[:, :-1] # (B, T-1)
        targets = x[:, 1:] # (B, T-1)
        
        logits = self(inputs) # (B, T-1, V)
        
        B, T, V = logits.shape
        
        logits = logits.reshape(B * T, V) # (B*T, V)
        targets = targets.reshape(B * T) # (B*T)
        
        return F.cross_entropy(logits, targets) # ignore_index = -100
    
    @torch.no_grad()
    def sample(self, n_samples: int, image_shape: tuple, temperature=1.0) -> torch.Tensor:
        device = next(self.parameters()).device
        C, H, W = image_shape
        total_pixels = C * H * W

        x = torch.zeros((n_samples, 1), dtype=torch.long, device=device)
        
        for _ in range(total_pixels):
            cond = x if x.size(1) <= self.pos_encoding.max_len else x[:, -self.pos_encoding.max_len:]

            logits = self(cond) # output shape: (B, T, vocab_size)

            next_token_logits = logits[:, -1, :] / temperature

            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1) # (B, 1)

            x = torch.cat((x, next_token), dim=1)

        output_seq = x[:, 1:] 
        
        output_image = output_seq.view(n_samples, H, W, C)

        return output_image.permute(0, 3, 1, 2)


class MMGPT(nn.Module):
    """Multimodal Transformer"""
    def __init__(self) -> None:
        super().__init__()
        raise NotImplementedError
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
    
    def loss(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
    
    def sample(self) -> torch.Tensor:
        raise NotImplementedError


# Pre-LN
class GPTBlock(nn.Module):
    """"""
    def __init__(self,
                d_model: int,
                num_heads: int,
                d_ff: int,
                theta: float | None = None,
                max_seq_len: int | None = None,
                device=None,
                dtype=None,
                ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff

        self.layer_norm1 = RMSNorm(d_model, **factory_kwargs)

        self.self_attn = RotaryCausalSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            theta=theta,
            max_seq_len=max_seq_len,
            rope_exist=True,
            **factory_kwargs,
            )

        self.layer_norm2 = RMSNorm(d_model, **factory_kwargs)
        self.ff = SwiGLU(d_model, d_ff)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Create token_positions
        seq_len = x.shape[1]
        token_positions = torch.arange(seq_len, device=x.device)

        attn_output = self.self_attn(self.layer_norm1(x),
                                         token_positions=token_positions
                                        )
        y = x + attn_output
        
        return y + self.ff(self.layer_norm2(y))


# Pre-LN
class IGPTBlock(nn.Module):
    def __init__(self,
                 d_model: int,
                 num_heads: int,
                 d_ff: int,
                 device=None,
                 dtype=None,
                 ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        
        self.layer_norm1 = LayerNorm(d_model, **factory_kwargs)
        
        self.self_attn = CausalSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            **factory_kwargs,
        )
        
        self.layer_norm2 = LayerNorm(d_model, **factory_kwargs)
        self.ff = FFN(d_model, d_ff)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_output = self.self_attn(self.layer_norm1(x))
        
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

def get_batch(dataset: npt.NDArray, batch_size: int, context_length: int, device: str):
    starting_idxs = torch.randint(len(dataset) - context_length, (batch_size,))
    x = torch.stack([torch.from_numpy((dataset[i:i+context_length]).astype(np.int64)) for i in starting_idxs])
    y = torch.stack([torch.from_numpy((dataset[i+1:i+1+context_length]).astype(np.int64)) for i in starting_idxs])
    # if "cuda" in device: passed
    if device.type == "cuda":
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x = x.to(device)
        y = y.to(device)
    return x, y