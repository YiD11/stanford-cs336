import logging

from typing import Tuple, List, Optional

import torch
from torch import (
    nn,
    Tensor, tensor,
    distributed as dist,
)

from dataclasses import dataclass

logger = logging.getLogger(__name__)

class DDPIndividualParameters(nn.Module):
    def __init__(self, module: nn.Module):
        super(DDPIndividualParameters, self).__init__()
        if dist.is_initialized():
            self.rank: int = dist.get_rank()
            self.word_size: int = dist.get_world_size()
        else:
            self.rank: int = 0
            self.word_size: int = 1
        self.module: nn.Module = module
        self._need_sync_grad: bool = False
        works = [dist.broadcast(param.data, src=0, async_op=True) for param in module.parameters()]
        for work in works: work.wait()

    def forward(self, *args, **kwargs):
        self._need_sync_grad = True
        return self.module(*args, **kwargs)

    def parameters(self, recurse=True):
        return self.module.parameters(recurse)

    def named_parameters(self, prefix="", recurse=True, remove_duplicate=True):
        return self.module.named_parameters(prefix, recurse, remove_duplicate)

    def state_dict(self, *args, **kwargs):
        return self.module.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        return self.module.load_state_dict(state_dict, strict, assign)

    def zero_grad(self, set_to_none=True):
        return self.module.zero_grad(set_to_none)

    def finish_gradient_synchronization(self):
        if not self._need_sync_grad:
            return

        for param in self.module.parameters():
            if not param.requires_grad or param.grad is None: continue
            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
            param.grad.data /= self.word_size

        self._need_sync_grad = False


class DDPOverlapIndividualParameters(nn.Module):
    def __init__(self, module: nn.Module):
        super(DDPOverlapIndividualParameters, self).__init__()
        self.module: nn.Module = module
        self.rank: int = dist.get_rank()
        self.word_size: int = dist.get_world_size()
        self._need_sync_grad: bool = False
        self._async_grad_works: List[Tuple[dist.Work, nn.Parameter, Tensor]] = []
        self._register_grad_hooks()

        works = [dist.broadcast(param.data, src=0, async_op=True) for param in module.parameters()]
        for work in works: work.wait()

    def forward(self, *args, **kwargs):
        self._need_sync_grad = True
        self._async_grad_works.clear()
        return self.module(*args, **kwargs)

    def _register_grad_hooks(self):
        def make_grad_hook(param: nn.Parameter):
            def hook(grad: Tensor):
                grad_c = grad.clone()
                work = dist.all_reduce(grad_c, op=dist.ReduceOp.SUM, async_op=True)
                self._async_grad_works.append((work, param, grad_c))

            return hook

        for param in self.module.parameters():
            if not param.requires_grad: continue
            param.register_hook(make_grad_hook(param))

    def finish_gradient_synchronization(self):
        if not self._need_sync_grad:
            return

        for work, param, grad_c in self._async_grad_works:
            work.wait()
            param.grad.copy_(grad_c).div_(self.word_size)

        self._need_sync_grad = False
        self._async_grad_works.clear()

    def zero_grad(self, set_to_none=True):
        self.module.zero_grad(set_to_none)
        self._async_grad_works.clear()
        self._need_sync_grad = False

    def parameters(self, recurse=True):
        return self.module.parameters(recurse)

    def named_parameters(self, prefix="", recurse=True, remove_duplicate=True):
        return self.module.named_parameters(prefix, recurse, remove_duplicate)

    def state_dict(self, *args, **kwargs):
        return self.module.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        return self.module.load_state_dict(state_dict, strict, assign)


class Bucket:
    def __init__(
            self,
            index: int = 0,
            num_param: int = 0,
            parameters: Optional[List[nn.Parameter]] = None,
    ):
        self.index = index
        self.parameters: List[nn.Parameter] = parameters if parameters is not None else []
        self.num_param = num_param
        self.num_param_backward_done = 0


class DDPBucketedIndividualParameters(nn.Module):
    def __init__(self, module: nn.Module, bucket_size_mb: float):
        super(DDPBucketedIndividualParameters, self).__init__()
        self.module: nn.Module = module
        self.rank: int = dist.get_rank()
        self.word_size: int = dist.get_world_size()

        self.bucket_size: float = int(bucket_size_mb * 1024 * 1024 / next(module.parameters()).dtype.itemsize)
        self._bucket_index: int = 0
        self._buckets: List[Bucket] = []

        self._need_sync_grad: bool = False
        self._async_grad_works: List[Tuple[dist.Work, List[Tensor], Tensor, Bucket]] = []

        self._broadcast_parameters()
        self._init_buckets()
        self._register_grad_hooks()

    def _init_buckets(self):
        bucket = Bucket(parameters=[], index=self._next_bucket_index())
        for param in reversed(list(self.module.parameters())):
            if not param.requires_grad: continue

            if bucket.num_param + param.numel() > self.bucket_size:
                self._buckets.append(bucket)
                bucket = Bucket(parameters=[], index=self._next_bucket_index())

            param.bucket_index = bucket.index
            bucket.parameters.append(param)
            bucket.num_param += param.numel()

        if bucket.num_param > 0:
            self._buckets.append(bucket)

    def _next_bucket_index(self) -> int:
        ret = self._bucket_index
        self._bucket_index += 1
        return ret

    def _register_grad_hooks(self):
        def make_grad_hook(param: nn.Parameter):
            def hook(grad: Tensor):
                bucket_index = getattr(param, 'bucket_index', None)
                if bucket_index is None: return grad
                
                bucket = self._buckets[bucket_index]
                bucket.num_param_backward_done += param.numel()
                if bucket.num_param == bucket.num_param_backward_done:
                    grads = [p.grad for p in bucket.parameters if p.grad is not None]
                    bucket.num_param_backward_done = 0
                    if not grads: return
                    flattened_grads = torch._utils._flatten_dense_tensors(grads)
                    work = dist.all_reduce(flattened_grads, op=dist.ReduceOp.SUM, async_op=True)
                    self._async_grad_works.append((work, grads, flattened_grads, bucket))
                return

            return hook

        for param in self.module.parameters():
            if not param.requires_grad: continue
            param.register_post_accumulate_grad_hook(make_grad_hook(param))

    def forward(self, *args, **kwargs):
        self._need_sync_grad = True
        return self.module.forward(*args, **kwargs)

    def finish_gradient_synchronization(self):
        if not self._need_sync_grad: return
        for work, grads, flattened_grads, bucket in self._async_grad_works:
            work.wait()
            unflattened_grads = torch._utils._unflatten_dense_tensors(flattened_grads, grads)
            for grad, new_grad in zip(grads, unflattened_grads):
                grad.copy_(new_grad).div_(self.word_size)
            bucket.num_param_backward_done = 0

        self._async_grad_works.clear()
        self._need_sync_grad = False

    def _broadcast_parameters(self):
        works = [dist.broadcast(param.data, src=0, async_op=True) for param in self.module.parameters()]
        for work in works: work.wait()

    def _reset_bucket(self):
        self._buckets.clear()
        self._bucket_index = 0
        self._init_buckets()
    
    def zero_grad(self, set_to_none=True):
        self.module.zero_grad(set_to_none)
        for bucket in self._buckets:
            bucket.num_param_backward_done = 0
        self._async_grad_works.clear()
        self._need_sync_grad = False

    def parameters(self, recurse=True):
        return self.module.parameters(recurse)

    def named_parameters(self, prefix="", recurse=True, remove_duplicate=True):
        return self.module.named_parameters(prefix, recurse, remove_duplicate)

    def state_dict(self, *args, **kwargs):
        return self.module.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        return self.module.load_state_dict(state_dict, strict, assign)