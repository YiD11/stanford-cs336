from typing import Optional, Callable, overload, Any, Iterable, Union, Type, List

import torch
from torch import (
    nn,
    distributed as dist,
    optim
)

from torch.optim.optimizer import StateDict


class ShardOptimizer(optim.Optimizer):
    def __init__(
            self,
            params: Union[Iterable[nn.Parameter], nn.Parameter],
            optimizer_cls: Type[optim.Optimizer],
            **kwargs,
    ):
        if isinstance(params, nn.Parameter): params = [params]
        self.rank = 0
        self.world_size = 1
        if dist.is_initialized():
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        self.optimizer_cls = optimizer_cls
        self.all_params = []
        self.handled_params = []
        self.kwargs = dict(**kwargs)
        super().__init__(params, {})

    @overload
    def step(self, closure: None = ...) -> None:
        ...

    @overload
    def step(self, closure: Callable[[], float]) -> float:
        ...

    def step(self, closure: Optional[Callable[[], float]] = None, **kwargs) -> Optional[float]:
        loss = None
        if closure is not None:
            loss = self.optimizer.step(closure, **kwargs)
        else:
            self.optimizer.step(**kwargs)
        self._broadcast_parameters()
        return loss

    def _broadcast_parameters(self):
        if dist.get_world_size() <= 1: return
        works = [
            dist.broadcast(param.data, src=i % self.world_size, async_op=True)
            for i, param in enumerate(self.all_params) if param.requires_grad
        ]
        for work in works: work.wait()

    def add_param_group(self, param_group: dict[str, Any]) -> None:
        super().add_param_group(param_group)
        params = param_group["params"]
        if isinstance(params, nn.Parameter):
            params = [params]
        start = len(self.all_params) if self.all_params else 0
        self.all_params.extend(param for param in params if param.requires_grad)
        end = len(self.all_params) if self.all_params else 0
        self.handled_params.extend(self.all_params[i] for i in range(start, end) if i % self.world_size == self.rank)
        self.optimizer = self.optimizer_cls(self.handled_params, **self.kwargs)

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.optimizer.zero_grad(set_to_none)

    def state_dict(self) -> StateDict:
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict: StateDict) -> None:
        self.optimizer.load_state_dict(state_dict)
