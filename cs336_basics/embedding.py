
import torch
import torch.nn as nn
from torch import Tensor

class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        weights: Tensor | None = None,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weights = nn.Parameter(
            torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype) if weights is None else weights.to(dtype, device)
        )

    def forward(self, token_ids: Tensor) -> Tensor:
        """
        Forward pass of the embedding layer.
        Args:
            indices (Int[Tensor, "..."]): Input tensor containing indices.
        Returns:
            Float[Tensor, "... d_model"]: Output tensor after applying the embedding transformation.
        """
        if not isinstance(token_ids, Tensor):
            token_ids = Tensor(token_ids)
        return self.weights[token_ids]