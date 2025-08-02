import os
from typing import Tuple, Any, Annotated, Optional

os.environ["TRITON_INTERPRET"] = "1"  # debugging

os.environ["TORCH_LOGS"] = "dynamic"
os.environ["TORCHDYNAMO_VERBOSE"] = "1"

import numpy as np

import torch
from torch import nn, Tensor, tensor

torch._dynamo.config.capture_scalar_outputs = True

import triton
import triton.language as tl

cdiv = lambda a, b: (a + b - 1) // b


@triton.jit
def flash_attention_forward_kernel(
        q_ptr: tl.tensor, k_ptr: tl.tensor, v_ptr: tl.tensor,
        out_ptr: tl.tensor, logsum_ptr: tl.tensor,
        q_batch_stride: tl.constexpr, q_seq_len_stride: tl.constexpr, q_dim_stride: tl.constexpr,
        k_batch_stride: tl.constexpr, k_seq_len_stride: tl.constexpr, k_dim_stride: tl.constexpr,
        v_batch_stride: tl.constexpr, v_seq_len_stride: tl.constexpr, v_dim_stride: tl.constexpr,
        out_batch_stride: tl.constexpr, out_seq_len_stride: tl.constexpr, out_dim_stride: tl.constexpr,
        stride_logsum_batch: tl.constexpr, stride_logsum_seq_len: tl.constexpr,

        q_seq_len: tl.constexpr, k_seq_len: tl.constexpr, dim: tl.constexpr, scale: float,
        q_tile_size: tl.constexpr, k_tile_size: tl.constexpr,
        is_causal=False  # whether to apply causal mask
):
    # load data
    i = tl.program_id(0)  # tile index of query
    b = tl.program_id(1)  # batch index of query
    q_block_ptr = tl.make_block_ptr(
        base=q_ptr + b * q_batch_stride, shape=(q_seq_len, dim), strides=(q_seq_len_stride, q_dim_stride),
        offsets=(i * q_tile_size, 0), block_shape=(q_tile_size, dim), order=(1, 0) # seq_len greater than dim
    )
    q = tl.load(q_block_ptr, boundary_check=(0, 1), padding_option='zero')  # load query tile
    k_block_ptr = tl.make_block_ptr(
        base=k_ptr + b * k_batch_stride, shape=(dim, k_seq_len), strides=(k_dim_stride, k_seq_len_stride),
        offsets=(0, 0), block_shape=(dim, k_tile_size), order=(0, 1)
    )
    v_block_ptr = tl.make_block_ptr(
        base=v_ptr + b * v_batch_stride, shape=(k_seq_len, dim), strides=(v_seq_len_stride, v_dim_stride),
        offsets=(0, 0), block_shape=(k_tile_size, dim), order=(1, 0)
    )
    out_block_ptr = tl.make_block_ptr(
        base=out_ptr + b * out_batch_stride, shape=(q_seq_len, dim), strides=(out_seq_len_stride, out_dim_stride),
        offsets=(i * q_tile_size, 0), block_shape=(q_tile_size, dim), order=(1, 0)
    )
    logsum_block_ptr = tl.make_block_ptr(
        base=logsum_ptr + b * stride_logsum_batch, shape=(q_seq_len,), strides=(stride_logsum_seq_len,),
        offsets=(i * q_tile_size,), block_shape=(q_tile_size,), order=(0,)
    )

    row_max = tl.zeros((q_tile_size,), dtype=tl.float32) - float('inf')  # row max for the tile
    l = tl.zeros((q_tile_size,), dtype=tl.float32)
    o = tl.zeros((q_tile_size, dim), dtype=tl.float32)
    for j in range(0, k_seq_len, k_tile_size):
        k = tl.load(k_block_ptr, boundary_check=(0, 1), padding_option='zero')  # (d, k_tile_size)
        v = tl.load(v_block_ptr, boundary_check=(0, 1), padding_option='zero')  # (k_tile_size, d)

        score = tl.dot(q, k) * scale  # (q_tile_size, k_tile_size)
        if is_causal:  # apply causal mask if needed
            q_tile_indices = tl.arange(0, q_tile_size) + i * q_tile_size  # indices for the query tile
            k_tile_indices = tl.arange(0, k_tile_size) + j  # indices for the key tile
            causal_mask = q_tile_indices[:, None] >= k_tile_indices[None, :]
            score += tl.where(causal_mask, 0.0, float('-1e6'))
        next_row_max = tl.maximum(row_max, tl.max(score, axis=1))  # (q_tile_size,)
        probs = tl.exp(score - next_row_max[:, None])  # (q_tile_size, k_tile_size)
        next_l = tl.exp(row_max - next_row_max) * l + tl.sum(probs,
                                                             axis=1)  # (q_tile_size.sum(axis=-1)  # (q_tile_size,)
        o = tl.exp(row_max - next_row_max)[:, None] * o + tl.dot(probs, v)  # (q_tile_size, d)

        # update
        row_max = next_row_max
        l = next_l
        # to next tile
        k_block_ptr = k_block_ptr.advance((0, k_tile_size))
        v_block_ptr = v_block_ptr.advance((k_tile_size, 0))

    o /= l[:, None]
    logsum = row_max + tl.log(l)
    tl.store(out_block_ptr, o.to(out_block_ptr.type.element_ty), boundary_check=(0, 1))
    tl.store(logsum_block_ptr, logsum.to(logsum_block_ptr.type.element_ty), boundary_check=(0,))


@triton.jit
def flash_attention_backward_kernel(
        q_ptr: tl.tensor, q_grad_ptr: tl.tensor, q_seq_len: tl.constexpr, q_tile_size: tl.constexpr,
        q_batch_stride: tl.constexpr, q_seq_len_stride: tl.constexpr, q_dim_stride: tl.constexpr,
        q_grad_batch_stride: tl.constexpr, q_grad_seq_len_stride: tl.constexpr, q_grad_dim_stride: tl.constexpr,

        k_ptr: tl.tensor, k_grad_ptr: tl.tensor, k_seq_len: tl.constexpr, k_tile_size: tl.constexpr,
        k_batch_stride: tl.constexpr, k_seq_len_stride: tl.constexpr, k_dim_stride: tl.constexpr,
        k_grad_batch_stride: tl.constexpr, k_grad_seq_len_stride: tl.constexpr, k_grad_dim_stride: tl.constexpr,

        v_ptr: tl.tensor, v_grad_ptr: tl.tensor,
        v_batch_stride: tl.constexpr, v_seq_len_stride: tl.constexpr, v_dim_stride: tl.constexpr,
        v_grad_batch_stride: tl.constexpr, v_grad_seq_len_stride: tl.constexpr, v_grad_dim_stride: tl.constexpr,

        out_ptr: tl.tensor, out_grad_ptr: tl.tensor,
        out_batch_stride: tl.constexpr, out_seq_len_stride: tl.constexpr, out_dim_stride: tl.constexpr,
        out_grad_batch_stride: tl.constexpr, out_grad_seq_len_stride: tl.constexpr, out_grad_dim_stride: tl.constexpr,

        logsum_ptr: tl.tensor,
        logsum_batch_stride: tl.constexpr, logsum_seq_len_stride: tl.constexpr,

        scale: float, dim: tl.constexpr, is_causal: bool = False
):
    b = tl.program_id(1)
    # backward key and value
    j = tl.program_id(0)
    k_block_ptr = tl.make_block_ptr(
        base=k_ptr + b * k_batch_stride, shape=(dim, k_seq_len), strides=(k_dim_stride, k_seq_len_stride),
        offsets=(0, j * k_tile_size), block_shape=(dim, k_tile_size), order=(0, 1)
    )
    k = tl.load(k_block_ptr, boundary_check=(0, 1), padding_option='zero')  # (dim, k_tile_size)
    v_block_ptr = tl.make_block_ptr(
        base=v_ptr + b * v_batch_stride, shape=(k_seq_len, dim), strides=(v_seq_len_stride, v_dim_stride),
        offsets=(j * k_tile_size, 0), block_shape=(k_tile_size, dim), order=(1, 0)
    )
    v = tl.load(v_block_ptr, boundary_check=(0, 1), padding_option='zero')  # (k_tile_size, dim)
    k_grad_block_ptr = tl.make_block_ptr(
        base=k_grad_ptr + b * k_grad_batch_stride, shape=(k_seq_len, dim),
        strides=(k_grad_seq_len_stride, k_grad_dim_stride),
        offsets=(j * k_tile_size, 0), block_shape=(k_tile_size, dim), order=(1, 0)
    )
    v_grad_block_ptr = tl.make_block_ptr(
        base=v_grad_ptr + b * v_grad_batch_stride, shape=(k_seq_len, dim),
        strides=(v_grad_seq_len_stride, v_grad_dim_stride),
        offsets=(j * k_tile_size, 0), block_shape=(k_tile_size, dim), order=(1, 0)
    )
    q_block_ptr = tl.make_block_ptr(
        base=q_ptr + b * q_batch_stride, shape=(q_seq_len, dim), strides=(q_seq_len_stride, q_dim_stride),
        offsets=(0, 0), block_shape=(q_tile_size, dim), order=(1, 0)
    )
    out_block_ptr = tl.make_block_ptr(
        base=out_ptr + b * out_batch_stride, shape=(q_seq_len, dim), strides=(out_seq_len_stride, out_dim_stride),
        offsets=(0, 0), block_shape=(q_tile_size, dim), order=(1, 0)
    )
    out_grad_block_ptr = tl.make_block_ptr(
        base=out_grad_ptr + b * out_grad_batch_stride, shape=(q_seq_len, dim),
        strides=(out_grad_seq_len_stride, out_grad_dim_stride),
        offsets=(0, 0), block_shape=(q_tile_size, dim), order=(1, 0)
    )
    logsum_block_ptr = tl.make_block_ptr(
        base=logsum_ptr + b * logsum_batch_stride, shape=(q_seq_len,), strides=(logsum_seq_len_stride,),
        offsets=(0,), block_shape=(q_tile_size,), order=(0,)
    )
    k_grad = tl.zeros((k_tile_size, dim), dtype=tl.float32)
    v_grad = tl.zeros((k_tile_size, dim), dtype=tl.float32)

    
    for i in tl.range(0, q_seq_len, q_tile_size):
        q = tl.load(q_block_ptr, boundary_check=(0, 1), padding_option='zero')  # (q_tile_size, dim)
        out = tl.load(out_block_ptr, boundary_check=(0, 1), padding_option='zero')
        out_grad = tl.load(out_grad_block_ptr, boundary_check=(0, 1), padding_option='zero')
        logsum = tl.load(logsum_block_ptr, boundary_check=(0,), padding_option='zero')

        d = tl.sum(out * out_grad, axis=1)  # (q_tile_size,)
        score = tl.dot(q, k) * scale
        if is_causal:
            q_tile_indices = tl.arange(0, q_tile_size) + i
            k_tile_indices = tl.arange(0, k_tile_size) + j * k_tile_size
            causal_mask = q_tile_indices[:, None] >= k_tile_indices[None, :]
            score += tl.where(causal_mask, 0.0, float('-1e6'))
        probs = tl.exp(score - logsum[:, None])  # (q_tile_size, k_tile_size)
        v_grad += tl.dot(probs.trans((1, 0)), out_grad)  # (k_tile_size, dim)
        probs_grad = tl.dot(out_grad, v.trans((1, 0)))  # (q_tile_size, k_tile_size)
        score_grad = probs * (probs_grad - d[:, None]) * scale  # (q_tile_size, k_tile_size)

        k_grad += tl.dot(score_grad.trans((1, 0)), q)
        # step to next tile
        q_block_ptr = q_block_ptr.advance((q_tile_size, 0))
        out_block_ptr = out_block_ptr.advance((q_tile_size, 0))
        out_grad_block_ptr = out_grad_block_ptr.advance((q_tile_size, 0))
        logsum_block_ptr = logsum_block_ptr.advance((q_tile_size,))

    tl.store(k_grad_block_ptr, k_grad.to(k_grad_block_ptr.type.element_ty), boundary_check=(0, 1))
    tl.store(v_grad_block_ptr, v_grad.to(v_grad_block_ptr.type.element_ty), boundary_check=(0, 1))

    # backward query
    i = tl.program_id(0)
    q_block_ptr = tl.make_block_ptr(
        base=q_ptr + b * q_batch_stride, shape=(q_seq_len, dim), strides=(q_seq_len_stride, q_dim_stride),
        offsets=(i * q_tile_size, 0), block_shape=(q_tile_size, dim), order=(1, 0)
    )
    q = tl.load(q_block_ptr, boundary_check=(0, 1), padding_option='zero')
    q_grad_block_ptr = tl.make_block_ptr(
        base=q_grad_ptr + b * q_grad_batch_stride,
        shape=(q_seq_len, dim), strides=(q_grad_seq_len_stride, q_grad_dim_stride),
        offsets=(i * q_tile_size, 0), block_shape=(q_tile_size, dim), order=(1, 0)
    )
    out_block_ptr = tl.make_block_ptr(
        base=out_ptr + b * out_batch_stride, shape=(q_seq_len, dim), strides=(out_seq_len_stride, out_dim_stride),
        offsets=(i * q_tile_size, 0), block_shape=(q_tile_size, dim), order=(1, 0)
    )
    out = tl.load(out_block_ptr, boundary_check=(0, 1), padding_option='zero')
    out_grad_block_ptr = tl.make_block_ptr(
        base=out_grad_ptr + b * out_grad_batch_stride, shape=(q_seq_len, dim),
        strides=(out_grad_seq_len_stride, out_grad_dim_stride),
        offsets=(i * q_tile_size, 0), block_shape=(q_tile_size, dim), order=(1, 0)
    )
    out_grad = tl.load(out_grad_block_ptr, boundary_check=(0, 1), padding_option='zero')
    logsum_block_ptr = tl.make_block_ptr(
        base=logsum_ptr + b * logsum_batch_stride, shape=(q_seq_len,), strides=(logsum_seq_len_stride,),
        offsets=(i * q_tile_size,), block_shape=(q_tile_size,), order=(0,)
    )
    logsum = tl.load(logsum_block_ptr, boundary_check=(0,), padding_option='zero')
    d = tl.sum(out * out_grad, axis=1)  # (q_tile_size,)

    k_block_ptr = tl.make_block_ptr(
        base=k_ptr + b * k_batch_stride, shape=(dim, k_seq_len), strides=(k_dim_stride, k_seq_len_stride),
        offsets=(0, 0), block_shape=(dim, k_tile_size), order=(0, 1)
    )
    v_block_ptr = tl.make_block_ptr(
        base=v_ptr + b * v_batch_stride, shape=(k_seq_len, dim), strides=(v_seq_len_stride, v_dim_stride),
        offsets=(0, 0), block_shape=(k_tile_size, dim), order=(1, 0)
    )
    q_grad = tl.zeros((q_tile_size, dim), dtype=tl.float32)
    for j in tl.range(0, k_seq_len, k_tile_size):
        k = tl.load(k_block_ptr, boundary_check=(0, 1), padding_option='zero')
        v = tl.load(v_block_ptr, boundary_check=(0, 1), padding_option='zero')
        score = tl.dot(q, k) * scale
        if is_causal:
            q_tile_indices = tl.arange(0, q_tile_size) + i * q_tile_size
            k_tile_indices = tl.arange(0, k_tile_size) + j
            causal_mask = q_tile_indices[:, None] >= k_tile_indices[None, :]
            score += tl.where(causal_mask, 0, float('-1e6'))
        probs = tl.exp(score - logsum[:, None])  # (q_tile_size, k_tile_size)
        probs_grad = tl.dot(out_grad, v.trans((1, 0)))  # (q_tile_size, k_tile_size)
        score_grad = probs * (probs_grad - d[:, None]) * scale  # (q_tile_size, k_tile_size)
        q_grad += tl.dot(score_grad, k.trans((1, 0)))

        k_block_ptr = k_block_ptr.advance((0, k_tile_size))
        v_block_ptr = v_block_ptr.advance((k_tile_size, 0))
    
    tl.store(q_grad_block_ptr, q_grad.to(q_grad_block_ptr.type.element_ty), boundary_check=(0, 1))


class FlashAttentionFusion(torch.autograd.Function):

    @staticmethod
    def forward(
            ctx,
            q: Tensor, k: Tensor, v: Tensor, is_causal: bool = False
    ):
        assert q.shape[-1] == k.shape[-1] == v.shape[-1], "Query, key, and value must have the same dimension size."
        assert q.dtype == k.dtype == v.dtype, "All tensors must have the same dtype"
        ctx.is_causal = is_causal
        ctx.q_tile_size = ctx.k_tile_size = 16
        ctx.dim = q.shape[-1]
        ctx.q_seq_len = q.shape[-2]
        ctx.k_seq_len = k.shape[-2]
        ctx.other_dims = q.shape[:-2]
        ctx.scale = 1 / (ctx.dim ** 0.5)
        n_b = int(np.prod(ctx.other_dims))  # number of batches
        q_batch_stride = int(np.prod(q.stride()[:-2]))
        k_batch_stride = int(np.prod(k.stride()[:-2]))
        v_batch_stride = int(np.prod(v.stride()[:-2]))

        out = torch.empty_like(q, dtype=q.dtype, device=q.device)
        out_batch_stride = int(np.prod(out.stride()[:-2]))
        logsum = torch.empty(q.shape[:-1], dtype=q.dtype, device=q.device)
        logsum_batch_stride = int(np.prod(logsum.stride()[:-1]))

        kernel_shape = (cdiv(ctx.q_seq_len, ctx.q_tile_size), n_b, 1)
        flash_attention_forward_kernel[kernel_shape](
            q_ptr=q, k_ptr=k, v_ptr=v,
            out_ptr=out, logsum_ptr=logsum,
            q_batch_stride=q_batch_stride, q_seq_len_stride=q.stride(-2), q_dim_stride=q.stride(-1),
            k_batch_stride=k_batch_stride, k_seq_len_stride=k.stride(-2), k_dim_stride=k.stride(-1),
            v_batch_stride=v_batch_stride, v_seq_len_stride=v.stride(-2), v_dim_stride=v.stride(-1),
            out_batch_stride=out_batch_stride, out_seq_len_stride=out.stride(-2), out_dim_stride=out.stride(-1),
            stride_logsum_batch=logsum_batch_stride, stride_logsum_seq_len=logsum.stride(-1),

            q_seq_len=q.shape[-2], k_seq_len=k.shape[-2], dim=ctx.dim, scale=ctx.scale,
            q_tile_size=ctx.q_tile_size, k_tile_size=ctx.k_tile_size, is_causal=is_causal
        )
        ctx.save_for_backward(logsum, q, k, v, out)
        return out

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        logsum, q, k, v, out = ctx.saved_tensors

        q_batch_stride = int(np.prod(q.stride()[:-2]))
        k_batch_stride = int(np.prod(k.stride()[:-2]))
        v_batch_stride = int(np.prod(v.stride()[:-2]))
        out_batch_stride = int(np.prod(out.stride()[:-2]))
        logsum_batch_stride = int(np.prod(logsum.stride()[:-1]))

        q_grad = torch.zeros_like(q, dtype=q.dtype, device=q.device)
        k_grad = torch.zeros_like(k, dtype=k.dtype, device=k.device)
        v_grad = torch.zeros_like(v, dtype=v.dtype, device=v.device)
        q_grad_batch_stride = int(np.prod(q_grad.stride()[:-2]))
        k_grad_batch_stride = int(np.prod(k_grad.stride()[:-2]))
        v_grad_batch_stride = int(np.prod(v_grad.stride()[:-2]))
        out_grad = grad_outputs[0]
        out_grad_batch_stride = int(np.prod(out_grad.stride()[:-2]))

        n_b = int(np.prod(ctx.other_dims))  # number of batches
        kernel_shape = (
            max(cdiv(ctx.k_seq_len, ctx.k_tile_size), cdiv(ctx.q_seq_len, ctx.q_tile_size)),
            n_b,
            1
        )

        flash_attention_backward_kernel[kernel_shape](
            q_ptr=q, q_grad_ptr=q_grad, q_seq_len=q.shape[-2], q_tile_size=ctx.q_tile_size,
            q_batch_stride=q_batch_stride, q_seq_len_stride=q.stride(-2), q_dim_stride=q.stride(-1),
            q_grad_batch_stride=q_grad_batch_stride,
            q_grad_seq_len_stride=q_grad.stride(-2), q_grad_dim_stride=q_grad.stride(-1),

            k_ptr=k, k_grad_ptr=k_grad, k_seq_len=k.shape[-2], k_tile_size=ctx.k_tile_size,
            k_batch_stride=k_batch_stride, k_seq_len_stride=k.stride(-2), k_dim_stride=k.stride(-1),
            k_grad_batch_stride=k_grad_batch_stride,
            k_grad_seq_len_stride=k_grad.stride(-2), k_grad_dim_stride=k_grad.stride(-1),

            v_ptr=v, v_grad_ptr=v_grad,
            v_batch_stride=v_batch_stride, v_seq_len_stride=v.stride(-2), v_dim_stride=v.stride(-1),
            v_grad_batch_stride=v_grad_batch_stride,
            v_grad_seq_len_stride=v_grad.stride(-2), v_grad_dim_stride=v_grad.stride(-1),

            out_ptr=out, out_grad_ptr=out_grad,
            out_batch_stride=out_batch_stride, out_seq_len_stride=out.stride(-2), out_dim_stride=out.stride(-1),
            out_grad_batch_stride=out_grad_batch_stride,
            out_grad_seq_len_stride=out_grad.stride(-2), out_grad_dim_stride=out_grad.stride(-1),

            logsum_ptr=logsum,
            logsum_batch_stride=logsum_batch_stride, logsum_seq_len_stride=logsum.stride(-1),
            scale=ctx.scale, dim=ctx.dim, is_causal=ctx.is_causal,
        )
        return q_grad, k_grad, v_grad, None


class FlashAttentionPytorch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q: Tensor, k: Tensor, v: Tensor, is_causal=False):
        ctx.tile_size = q_tile_size = k_tile_size = 32
        dim = q.shape[-1]
        scale = 1 / (dim ** 0.5)
        out, logsum = flash_attention_forward(
            q, k, v,
            scale,
            q_tile_size=q_tile_size, k_tile_size=k_tile_size,
            is_causal=is_causal,
            device=q.device,
        )
        ctx.save_for_backward(logsum, q, k, v, out)
        return out

    @staticmethod
    def backward(
            ctx, *grad_output: Tensor
    ) -> tuple[Optional[Tensor], Optional[Tensor], Optional[Tensor], Optional[Tensor]]:
        logsum, q, k, v, out = ctx.saved_tensors
        scale = 1 / (q.shape[-1] ** 0.5)
        d_out = grad_output[0]
        grad_q, grad_k, grad_v = flash_attention_backward(
            q, k, v, out, logsum,
            scale=scale,
            d_out=d_out,
            device=q.device,
        )
        return grad_q, grad_k, grad_v, None


@torch.compile
def flash_attention_forward(
        q: Tensor, k: Tensor, v: Tensor,
        scale: float,
        q_tile_size: int = 16, k_tile_size: int = 16,
        is_causal: bool = False,
        dtype=torch.float32,
        device=None
) -> Tuple[Tensor, Tensor]:
    assert q.shape[-1] == k.shape[-1] == v.shape[-1], "Query, key, and value must have the same last dimension size."
    assert k.shape[-2:] == v.shape[-2:], "Key and value must have the same last two dimensions."
    q_seq_len, dim = q.shape[-2:]
    k_seq_len, _ = k.shape[-2:]
    assert q_seq_len % q_tile_size == 0, "Query sequence length must be divisible by tile size."
    assert k_seq_len % k_tile_size == 0, "Key sequence length must be divisible by tile size."
    q_tile_num = q_seq_len // q_tile_size
    k_tile_num = k_seq_len // k_tile_size
    q = q.to(device, dtype)
    k = k.to(device, dtype)
    v = v.to(device, dtype)

    q_tiles = q.reshape((*q.shape[:-2], q_tile_num, q_tile_size, dim))  # (..., q_tile_num, q_tile_size, dim)
    k_tiles = k.reshape((*k.shape[:-2], k_tile_num, k_tile_size, dim))  # (..., k_tile_num, k_tile_size, dim)
    v_tiles = v.reshape((*v.shape[:-2], k_tile_num, k_tile_size, dim))  # (..., k_tile_num, k_tile_size, dim)
    out_tiles = torch.zeros_like(q_tiles, dtype=dtype, device=device)  # (..., q_tile_num, q_tile_size, dim)
    row_max = torch.empty(
        (*q.shape[:-2], q_tile_num, q_tile_size),
        dtype=dtype, device=device
    ).fill_(-float('inf'))  # (..., q_tile_num, q_tile_size)
    l = torch.zeros(
        (*q.shape[:-2], q_tile_num, q_tile_size),
        dtype=dtype, device=device
    )  # (..., q_tile_num, q_tile_size)

    for i in torch.arange(k_tile_num):
        k_tile = k_tiles[..., i, None, :, :]  # (..., 1, k_tile_size, dim), 1 for broadcasting
        q_tile_indices = (
                torch.arange(0, q_seq_len, q_tile_size, device=device)[:, None]
                + torch.arange(q_tile_size, device=device)[None, :]
        )[..., None]  # (..., q_tile_num, q_tile_size, 1)
        k_tile_indices = (torch.arange(0, k_tile_size, device=device) + i * k_tile_size)[None, :]  # (1, k_tile_size)
        scores = (q_tiles @ k_tile.transpose(-1, -2)) * scale  # (..., q_tile_num, q_tile_size, k_tile_size)
        if is_causal:
            causal_mask = q_tile_indices < k_tile_indices  # (..., q_tile_num, q_tile_size, k_tile_size)
            scores += torch.where(causal_mask, -1e6, 0)  # apply causal mask
        next_row_max = torch.maximum(row_max, scores.max(dim=-1).values)  # (..., q_tile_num, q_tile_size)
        probs = (scores - next_row_max[..., None]).exp()  # (..., q_tile_num, q_tile_size, k_tile_size)
        next_l = (row_max - next_row_max).exp() * l + probs.sum(dim=-1)  # (..., q_tile_num, q_tile_size)
        out_tiles = (row_max - next_row_max).exp()[..., None] * out_tiles + probs @ v_tiles[..., i, None, :, :]

        # update
        row_max = next_row_max
        l = next_l

    out = (out_tiles / l[..., None]).reshape(q.shape)  # (..., q_seq_len, dim)
    logsum = (row_max + l.log()).reshape(q.shape[:-1])  # (..., q_seq_len)
    return out, logsum


@torch.compile
def flash_attention_backward(
        q: Tensor, k: Tensor, v: Tensor,
        out: Tensor, logsum: Tensor,
        scale: float,
        d_out: Annotated[Tensor, "derivative of output"],
        q_tile_size: int = 16, k_tile_size: int = 16,
        is_causal: bool = False,
        dtype=torch.float32,
        device=None
) -> Tuple[Tensor, Tensor, Tensor]:
    assert q.shape[-2:] == out.shape[-2:] == d_out.shape[
        -2:], "Query, output, and derivative of output must have the same last two dimensions."
    assert k.shape[-2:] == v.shape[-2:], "Key and value must have the same last two dimensions."
    assert q.shape[-1] == k.shape[-1] == v.shape[-1]
    q_seq_len, dim = q.shape[-2:]
    k_seq_len, _ = k.shape[-2:]
    assert q_seq_len % q_tile_size == 0, "Query sequence length must be divisible by tile size."
    assert k_seq_len % k_tile_size == 0, "Key sequence length must be divisible by tile size."
    q = q.to(device, dtype)
    k = k.to(device, dtype)
    v = v.to(device, dtype)
    d_out = d_out.to(device, dtype)

    d = (d_out * out).sum(dim=-1)  # (batch, seq_len)
    q_tile_num = q_seq_len // q_tile_size
    q_tiles = q.reshape((*q.shape[:-2], q_tile_num, q_tile_size, dim))  # (batch, q_tile_num, q_tile_size, dim)
    d_out_tiles = d_out.reshape((*q.shape[:-2], q_tile_num, q_tile_size, dim))  # (batch, q_tile_num, q_tile_size, dim)
    logsum_tiles = logsum.reshape((*q.shape[:-2], q_tile_num, q_tile_size))  # (batch, q_tile_num, q_tile_size)
    d_tiles = d.reshape((*q.shape[:-2], q_tile_num, q_tile_size))  # (batch, q_tile_num, q_tile_size)

    k_tile_num = k_seq_len // k_tile_size
    k_tiles = k.reshape((*k.shape[:-2], k_tile_num, k_tile_size, dim))  # (batch, k_tile_num, k_tile_size, dim)
    v_tiles = v.reshape((*v.shape[:-2], k_tile_num, k_tile_size, dim))  # (batch, k_tile_num, k_tile_size, dim)

    q_grad = torch.zeros_like(q, dtype=dtype, device=device)
    k_grad = torch.zeros_like(k, dtype=dtype, device=device)
    v_grad = torch.zeros_like(v, dtype=dtype, device=device)

    for i in torch.arange(0, k_tile_num):
        k_tile = k_tiles[..., i, None, :, :]  # (batch, 1, k_tile_size, dim), 1 for broadcasting
        v_tile = v_tiles[..., i, None, :, :]  # (batch, 1, k_tile_size, dim)

        scores = (q_tiles @ k_tile.transpose(-1, -2)) * scale  # (batch, q_tile_num, q_tile_size, k_tile_size)
        probs = (scores - logsum_tiles[..., None]).exp()  # (batch, q_tile_num, q_tile_size, k_tile_size)
        d_probs = d_out_tiles @ v_tile.transpose(-1, -2)  # (batch, q_tile_num, q_tile_size, k_tile_size)
        d_scores = probs * (d_probs - d_tiles[..., None]) * scale  # (batch, q_tile_num, q_tile_size, k_tile_size)
        q_grad += (d_scores @ k_tile).reshape((*q.shape[:-1], dim))

        k_grad[
            ..., i * k_tile_size: (i + 1) * k_tile_size, :
        ] += (d_scores.transpose(-1, -2) @ q_tiles).sum(dim=-3)  # (batch, k_tile_size, dim)

        v_grad[
            ..., i * k_tile_size: (i + 1) * k_tile_size, :
        ] += (probs.transpose(-1, -2) @ d_out_tiles).sum(dim=-3)  # (batch, k_tile_size, dim)

    return (q_grad, k_grad, v_grad)


def test_timing_flash_forward_backward():
    n_heads = 16
    d_head = 64
    sequence_length = 16384
    q, k, v = torch.randn(
        3, n_heads, sequence_length, d_head, device='cuda', dtype=torch.bfloat16, requires_grad=True
    )
    flash = torch.compile(FlashAttentionPytorch.apply)

    def flash_forward_backward():
        o = flash(q, k, v, True)
        loss = o.sum()
        loss.backward()

    results = triton.testing.do_bench(flash_forward_backward, rep=10000, warmup=1000)
    print(results)