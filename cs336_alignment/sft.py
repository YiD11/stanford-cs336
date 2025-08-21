import torch

def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Get the entropy of the logits (i.e., entropy of the final dimension)."""
    log_probs = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
    probs = torch.exp(log_probs)
    return -torch.sum(probs * log_probs, dim=-1)

def get_response_log_probs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool,
) -> dict[str, torch.Tensor]:
    """Get the conditional log-probs of the response given the prompt,
        and optionally the entropy of the next token predictions.

    Args:
        model: PreTrainedModel, the model to score.
        input_ids: torch.Tensor of shape (batch_size, sequence_length):
            the tokenized prompt and output.
        labels: torch.Tensor of shape (batch_size, sequence_length):
            shifted input_ids.
        return_token_entropy: bool, whether to return the entropy of the
            next token predictions.

    Returns:
        dict[str, torch.Tensor]:
            "log_probs": torch.Tensor of shape (batch_size, sequence_length):
                the conditional log-probs of the response given the prompt.
                Note that we have not masked out the token indices corresponding
                to the prompt or padding; that is done in the train loop.
            "token_entropy": Optional[torch.Tensor] of shape (batch_size, sequence_length):
                the entropy of the next token predictions. As with the log-probs,
                we have not masked out the token indices corresponding to the prompt
                or padding; that is done in the train loop.
    """
    labels = labels.unsqueeze(-1)
    output = model(input_ids=input_ids)
    logits = output.logits
    log_probs = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
    probs = torch.exp(log_probs)
    token_log_probs = torch.take_along_dim(log_probs, labels, dim=-1)
    ret = {
        "log_probs": token_log_probs[..., 0], 
    }
    if return_token_entropy:
        token_entropy = -torch.sum(probs * log_probs, dim=-1)
        ret["token_entropy"] = token_entropy
    return ret

def masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None = None,
    normalize_constant: float = 1.0,
) -> torch.Tensor:
    """Sum over a dimension and normalize by a constant,
    considering only the elements with mask value 1.

    Args:
        tensor: torch.Tensor, the tensor to sum and normalize.
        mask: torch.Tensor, the mask. We only consider elements
            with mask value 1.
        dim: int | None, the dimension to sum along before
            normalization. If None, sum over all dimensions.
        normalize_constant: float, the constant to divide by
            for normalization.

    Returns:
        torch.Tensor, the normalized sum, where masked elements
            (mask=0) don't contribute to the sum.
    """
    masked_tensor = tensor * mask
    if dim is None:
        masked_sum = masked_tensor.sum()
    else:
        masked_sum = masked_tensor.sum(dim=dim)
    normalized_result = masked_sum / normalize_constant
    return normalized_result


def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: int | None = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the policy gradient loss and backprop its gradients for a microbatch.
    """
    normed_policy_log_probs = masked_normalize(policy_log_probs, response_mask, normalize_constant=normalize_constant)
    loss = -normed_policy_log_probs
    if gradient_accumulation_steps > 1:
        loss = loss / gradient_accumulation_steps / 2
    loss.backward()
    metadata = {
        "loss": loss.detach(),
        "normed_policy_log_probs": normed_policy_log_probs.detach(),
    }
    
    return loss, metadata
    