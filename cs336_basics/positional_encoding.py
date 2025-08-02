import torch
from torch import nn, Tensor

class RotaryPositionalEmbedding(nn.Module):
    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        
        # comput the positional encodings
        freq = theta ** (torch.arange(0, d_k, 2) / d_k)
        inv_freq = 1.0 / freq
        pos = torch.arange(0, max_seq_len, device=device)
        pos_freqs = torch.outer(pos, inv_freq)
        self.register_buffer("cos", pos_freqs.cos(), persistent=False)
        self.register_buffer("sin", pos_freqs.sin(), persistent=False)
    
    def forward(self, x: Tensor, token_positions: Tensor) -> Tensor:
        fir = x[..., 0::2]
        sec = x[..., 1::2]
        cos = self.cos[token_positions]
        sin = self.sin[token_positions]
        out = torch.zeros_like(x)
        out[..., 0::2] = fir * cos - sec * sin
        out[..., 1::2] = sec * cos + fir * sin
        
        return out
        