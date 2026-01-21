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
    "GPT",
    "IGPT",
    "MMGPT",
    "compute_lr",
    "gradient_cliping",
    "get_batch",
]

class GPT(nn.Module):
    """TextGPT"""
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
        x = x.to(next(self.parameters()).device)
        
        nll = F.cross_entropy(self(x[:, :-1]), x[:, 1:], reduction='none')
        mask = (x[:, :-1] != self.eos_ind).float()
        masked_nll = nll * mask
        loss = masked_nll.sum() / mask.sum()
        return loss
    
    def sample(self,
        x: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        eos_token_id: int | None = None,) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
            
        original_sequence_length = x.size(-1)

        for _ in range(max_new_tokens):

            # Take the last `context_length` tokens if the input is
            # beyond the model's context length
            x = x[:, -self.context_length :] if x.size(1) > self.context_length else x

            # Get the logits from the model
            logits = self.forward(x)

            # Take the logits for the next token
            next_token_logits = logits[:, -1]

            # apply temperature scaling
            temperature_scaled_next_token_logits = next_token_logits / temperature

            # If top-k is provided, take the tokens with the highest score
            if top_k:
                topk_values, _ = torch.topk(
                    temperature_scaled_next_token_logits,
                    min(top_k, temperature_scaled_next_token_logits.size(-1)),
                )

                # Get the score of the kth item that we kept---items with lower scores should be masked.
                threshold = topk_values[:, -1]
                topk_mask = temperature_scaled_next_token_logits < threshold
                temperature_scaled_next_token_logits.masked_fill(topk_mask, float("-inf"))
            next_token_probabilities = F.softmax(temperature_scaled_next_token_logits, dim=-1)
            next_token_id = torch.multinomial(next_token_probabilities, 1)

            # End generation if we see the EOS token ID
            if eos_token_id is not None and next_token_id.item() == eos_token_id:
                break
            x = torch.cat((x, next_token_id), dim=-1)

        new_token_ids = x[:, original_sequence_length:]
        return new_token_ids
    

class IGPT(nn.Module):
    """ImageGPT"""
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
        
        self.max_seq_len = max_seq_len
        
        self.embedding = Embedding(num_embeddings=vocab_size, embedding_dim=d_model, **factory_kwargs)
        
        self.pos = nn.Parameter(torch.zeros(1, self.max_seq_len, d_model))

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
        x = self.embedding(x)
        
        x = x + self.pos[:, :x.size(1)]
        
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
            cond = x if x.size(1) <= self.max_seq_len else x[:, -self.max_seq_len:]

            logits = self(cond) # output shape: (B, T, vocab_size)

            next_token_logits = logits[:, -1, :] / temperature

            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1) # (B, 1)

            x = torch.cat((x, next_token), dim=1)

        output_seq = x[:, 1:] 
        
        output_image = output_seq.view(n_samples, H, W, C)

        return output_image.float()


class MMGPT(nn.Module):
    """Multimodal GPT"""
    def __init__(self,
                 image_shape: tuple[Literal[20], Literal[20], Literal[1, 3]] | tuple[Literal[28], Literal[28], Literal[1, 3]],
                 n_colors: int,
                 vocab_size: int,
                 context_length: int,
                 d_model: int, 
                 num_heads: int,
                 kernel_size: int = 7,
                 ) -> None:
        super().__init__()
        self.image_shape = image_shape
        
        self.mmgpt_block = MMGPTBlock(
            d_model=d_model,
            num_heads=num_heads,
            n_dims=1,
        )
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


class MMGPTBlock(nn.Module):
    def __init__(self,
                 d_model: int = 128,
                 num_heads: int = 4,
                 n_dims: int = 1,
                 device=None,
                 dtype=None,
                 ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self._self_attn = MultimodalCausalSelfAttention(
            in_channels=d_model,
            embed_channels=d_model,
            out_channels=d_model,
            num_heads=num_heads,
            n_dims=n_dims,
            **factory_kwargs,
        )
        
        mod = nn.Conv1d if n_dims == 1 else nn.Conv2d if n_dims == 2 else nn.Conv3d
        self._ff = nn.Sequential(
            mod(d_model, int(4.*d_model), 1, 1, 0),
            nn.GELU(),
            mod(int(4.*d_model), d_model, 1, 1, 0)
        )
        
        self._layer_norm = nn.LayerNorm(d_model, **factory_kwargs)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_output = self._self_attn(x)
        
        y = x + attn_output
        
        return y + self._ff(self._layer_norm(y))

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