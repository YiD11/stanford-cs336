import importlib.metadata

__version__ = importlib.metadata.version("cs336_basics")

from .BPETokenizer import *

from .pretokenization import pretokenize, run_train_bpe

from .linear import (
    Linear,
    Embedding,
    SwiGLU,
)

from .embedding import Embedding

from .norm import (
    RMSNorm,
)

from .positional_encoding import RotaryPositionalEmbedding

from .activation import (
    silu,
)

from .attention import (
    scaled_dot_product_attention,
    MultiheadSelfAttention,
    MultiheadSelfAttentionWithRoPE
)

from .layer import (
    TransformerBlock,
    TransoformerLM,
)

from .loss import (
    cross_entropy_loss,
)

from .optimizer import (
    AdamW,
)

from .scheduler import (
    get_lr_cosine_schedule,
)

from .data import (
    get_batch,
)

from .serialization import (
    save_checkpoint,
    load_checkpoint,
)

from .utils import (
    gradient_clipping,
)