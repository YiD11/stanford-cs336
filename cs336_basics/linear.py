import torch
import torch.nn as nn
from torch import Tensor


class Linear(nn.Module):
    def __init__(
        self,
        d_in: int,
        d_out: int,
        weight: Tensor | None = None,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.weight = nn.Parameter(
            torch.randn(d_out, d_in, device=device, dtype=dtype) if weight is None else weight.to(dtype, device)
        )
        
    def forward(self, x: Tensor) -> Tensor:
        return torch.matmul(x, self.weight.T)

class Embedding(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        weight: Tensor | None = None,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.weight = nn.Parameter(
            torch.randn(vocab_size, embedding_dim, device=device, dtype=dtype) if weight is None else weight.to(dtype, device)
        )
    
    def forward(self, x: Tensor) -> Tensor:
        return self.weight[x]

class SwiGLU(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        w1_weight: Tensor | None = None,
        w2_weight: Tensor | None = None,
        w3_weight: Tensor | None = None,
    ):
        super().__init__()
        self.w1 = Linear(
            d_in=d_model,
            d_out=d_ff,
            weight=w1_weight,
        )
        self.w2 = Linear(
            d_in=d_ff,
            d_out=d_model,
            weight=w2_weight,
        )
        self.w3 = Linear(
            d_in=d_model,
            d_out=d_ff,
            weight=w3_weight,
        )
    
    def forward(self, x: Tensor) -> Tensor:
        x1 = self.w1(x)
        x3 = self.w3(x)
        silu = x1 * torch.sigmoid(x1)
        ret = self.w2(silu * x3)
        return ret