import torch
import torch.nn as nn
from torch import Tensor

class RMSNorm(nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        weight: Tensor | None = None,
        dtype=None,
        device=None,
    ):
        super().__init__()
        self.eps = eps
        self.dtype = dtype or torch.float32
        self.weight = nn.Parameter(
            torch.ones(d_model, device=device, dtype=self.dtype) if weight is None else weight.to(dtype=self.dtype, device=device)
        )
    
    def forward(self, x: Tensor) -> Tensor:
        x_dtype = x.dtype
        rms: Tensor = (
            x.square().mean(axis=-1, keepdims=True) + self.eps
        ).sqrt()
        ret = x / rms * self.weight
        return ret.to(dtype=x_dtype)
            
        