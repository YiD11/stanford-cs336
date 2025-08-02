import torch
from torch import Tensor, nn

def silu(x: Tensor) -> Tensor:
    return x * torch.sigmoid(x)