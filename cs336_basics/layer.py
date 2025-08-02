import torch
from torch import Tensor, nn

from .attention import MultiheadSelfAttention, MultiheadSelfAttentionWithRoPE

from .norm import RMSNorm

from .linear import Linear, SwiGLU, Embedding

class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int = None,
        max_seq_len: int = 2048,
        theta: float = 10000.0,
    ):
        super().__init__()
        self.ln1 = RMSNorm(d_model, dtype=torch.float32)
        self.ln2 = RMSNorm(d_model, dtype=torch.float32)

        self.attn = MultiheadSelfAttentionWithRoPE(
            d_model=d_model,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            theta=theta,
        )

        self.ffn = SwiGLU(d_model=d_model, d_ff=d_ff)
    
    def forward(self, x: Tensor) -> Tensor:
        h = self.attn(self.ln1(x)) + x
        ret = self.ffn(self.ln2(h)) + h
        return ret

class TransoformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        weights: dict[str, Tensor],
        # in_indices: Int[Tensor, " batch_size sequence_length"],
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.layers = nn.ModuleList(
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                max_seq_len=context_length,
                theta=rope_theta,
            ) for _ in range(num_layers)
        )
        self.ln_final = RMSNorm(d_model, dtype=torch.float32)
        self.token_embeddings = Embedding(vocab_size, d_model)
        self.lm_head = Linear(d_model, vocab_size)
    
    def forward(self, in_indices: Tensor) -> Tensor:
        x = self.token_embeddings(in_indices)
        for layer in self.layers:
            x = layer(x)
        x = self.ln_final(x)
        x = self.lm_head(x)
        return x