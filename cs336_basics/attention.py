import torch
from torch import Tensor, nn

from .linear import Linear
from .positional_encoding import RotaryPositionalEmbedding

def scaled_dot_product_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    mask: Tensor | None = None,
) -> Tensor:
    d_k = Tensor([k.shape[-1]])
    attention_matrix = q @ k.transpose(-2, -1) / torch.sqrt(d_k)
    if mask is not None:
        attention_matrix = attention_matrix.masked_fill(mask == 0, float('-inf'))
    attention_weights = torch.softmax(attention_matrix, dim=-1)
    output = torch.matmul(attention_weights, v)
    return output

class MultiheadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        q_proj_weight: Tensor | None = None,
        k_proj_weight: Tensor | None = None,
        v_proj_weight: Tensor | None = None,
        o_proj_weight: Tensor | None = None,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_k = d_model // num_heads
        self.q_proj = Linear(d_model, num_heads * self.d_k, q_proj_weight)
        self.k_proj = Linear(d_model, num_heads * self.d_k, k_proj_weight)
        self.v_proj = Linear(d_model, num_heads * self.d_k, v_proj_weight)
        self.output_proj = Linear(num_heads * self.d_k, d_model, o_proj_weight)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        batch_size, seq_len, _ = x.size()
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        if mask is None:
            mask = torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool).tril()
        output = scaled_dot_product_attention(q, k, v, mask)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.output_proj(output)

class MultiheadSelfAttentionWithRoPE(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int,
        theta: float,
        q_proj_weight: Tensor | None = None,
        k_proj_weight: Tensor | None = None,
        v_proj_weight: Tensor | None = None,
        o_proj_weight: Tensor | None = None,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.rope = RotaryPositionalEmbedding(theta=theta, d_k=d_model // num_heads, max_seq_len=max_seq_len)
        self.d_k = d_model // num_heads
        self.q_proj = Linear(d_model, num_heads * self.d_k, q_proj_weight)
        self.k_proj = Linear(d_model, num_heads * self.d_k, k_proj_weight)
        self.v_proj = Linear(d_model, num_heads * self.d_k, v_proj_weight)
        self.output_proj = Linear(num_heads * self.d_k, d_model, o_proj_weight)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        batch_size, seq_len, _ = x.size()
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        token_positions = torch.arange(seq_len, device=x.device)
        q_rope = self.rope(q, token_positions)
        k_rope = self.rope(k, token_positions)

        if mask is None:
            mask = torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool).tril()
        output = scaled_dot_product_attention(q_rope, k_rope, v, mask)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.output_proj(output)
