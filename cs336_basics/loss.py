import torch
from torch import Tensor, nn

def cross_entropy_loss(
    logits: Tensor,
    targets: Tensor,
    ignore_index: int = -100,
) -> Tensor:
    logits = logits.to(torch.float64)
    
    # Create mask for valid targets (not ignore_index)
    valid_mask = targets != ignore_index
    
    # Apply log-softmax for numerical stability
    log_probs = torch.log_softmax(logits, dim=-1)
    
    # Gather log probabilities for target classes
    target_log_probs = log_probs.gather(dim=-1, index=targets[..., None])[..., 0]
    
    # Apply mask and compute mean loss only over valid targets
    masked_loss = target_log_probs * valid_mask
    loss = -masked_loss.sum() / valid_mask.sum().clamp(min=1)
    
    return loss

    