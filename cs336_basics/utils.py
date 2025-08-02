from typing import Iterable
import torch
from torch import nn, Tensor

def gradient_clipping(
    parameters: Iterable[torch.nn.Parameter],
    max_l2_norm: float
):
    '''
    Given the gradient (for all parameters) g, we compute its l2-norm. If this norm is less than a maximum value, then we leave g as is; otherwise, we scale g down by a factor.
    '''
    params = list(parameters)
    grad_norm = torch.stack([p.grad for p in params if p.grad is not None]).norm(2)
    if grad_norm > max_l2_norm:
        factor = torch.clamp(max_l2_norm / (grad_norm + 1e-6), max=1.0)
        for p in params:
            if p.grad is not None:
                p.grad.mul_(factor)