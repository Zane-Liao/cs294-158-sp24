from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union

from jaxtyping import Float, Int

from deepul.models.attention.utils import MultiHeadSelfAttention
from deepul.models.modules.layers import RMSNorm, SwiGLU

__all__ = [
    "TransformerVM",
    "TransformerImage",
    "TransformerMM",
]

class TransformerVM(nn.Module):
    """"""
    def __init__(self,
                vocab_size: int,
                context_length: int,
                num_layers: int,
                d_model: int, 
                num_heads: int,
                d_ff: int, 
                rope_theta: float,
                ):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.num_layers = num_layers
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
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
        self.lm_head = nn.Linear(in_features=d_model, out_features=vocab_size)

    def forward(self, x: Int[torch.Tensor, "... sequence_length"]) -> Float[torch.Tensor, "... sequence_length vocab_size"]:
        _, sequence_length = x.size()

        x = self.embedding(x)
        
        for layer in self.layers:
            x = layer(x)
            
        x = self.ln_final(x)

        return self.lm_head(x)
    
    def loss(self) -> torch.Tensor:
        raise NotImplementedError
    
    def samples(self) -> torch.Tensor:
        raise NotImplementedError
    

class TransformerImage(nn.Module):
    def __init__(self):
        super().__init__()
        raise NotImplementedError
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
    
    def loss(self) -> torch.Tensor:
        raise NotImplementedError
    
    def samples(self) -> torch.Tensor:
        raise NotImplementedError


class TransformerMM(nn.Module):
    """Multimodal Transformer"""
    def __init__(self):
        super().__init__()
        raise NotImplementedError
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
    
    def loss(self) -> torch.Tensor:
        raise NotImplementedError
    
    def samples(self) -> torch.Tensor:
        raise NotImplementedError
    

class TransformerBlock(nn.Module):
    """"""
    def __init__(self,
                d_model: int,
                num_heads: int,
                d_ff: int,
                theta: float | None = None,
                max_seq_len: int | None = None,
                device=None,
                dtype=None,
                ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff

        self.rms_norm1 = RMSNorm(d_model, **factory_kwargs)
        # Test Case: MultiHeadSelfAttention <==> FlashAttention
        self.self_attn = MultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            theta=theta,
            max_seq_len=max_seq_len,
            rope_exist=True,
            **factory_kwargs,
            )

        self.rms_norm2 = RMSNorm(d_model, **factory_kwargs)
        self.ff = SwiGLU(d_model, d_ff)
    
    def forward(self,
                x: torch.Tensor
                ):
        # Create token_positions
        seq_len = x.shape[1]
        token_positions = torch.arange(seq_len, device=x.device)

        attn_output = self.self_attn(self.rms_norm1(x),
                                         token_positions=token_positions
                                        )
        y = x + attn_output
        
        return y + self.ff(self.rms_norm2(y))