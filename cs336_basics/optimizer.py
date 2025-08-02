from typing import Callable, Tuple, Union
import math
import torch
from torch import Tensor, nn, optim

class AdamW(optim.Optimizer):
    def __init__(
        self,
        params,
        betas: Tuple[Union[float, Tensor], Union[float, Tensor]] = (0.9, 0.999),
        weight_decay: float = 0.01,
        eps: float = 1e-8,
        lr: Union[float, Tensor] = 1e-3
    ):
        defaults = {"lr": lr}
        super().__init__(params, defaults)
        self.betas = betas
        self.weight_decay = weight_decay
        self.eps = eps
        self.lr = lr

    def step(self, closure: Callable | None = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None: continue
                # init
                grad = p.grad
                state = self.state.get(p, {})
                t: int = state.get('t', 1)
                moment1: Tensor = state.get('moment1', torch.zeros_like(p))
                moment2: Tensor = state.get('moment2', torch.zeros_like(p))
                beta1, beta2 = self.betas

                # calc new moments
                new_moment1: Tensor = beta1 * moment1 + (1 - beta1) * grad
                new_moment2: Tensor = beta2 * moment2 + (1 - beta2) * grad ** 2
                
                # update state
                state['moment1'] = new_moment1
                state['moment2'] = new_moment2
                state['t'] = t + 1
                self.state[p] = state
                
                # calc learning rate
                adjusted_lr = self.lr * torch.sqrt((Tensor([1]) - beta2 ** t)) / (Tensor([1]) - beta1 ** t)
                # update parameters
                p.data = p.data - adjusted_lr * new_moment1 / (new_moment2.sqrt() + self.eps)
                if self.weight_decay != 0:
                    p.data = p.data - self.lr * self.weight_decay * p.data

        return loss