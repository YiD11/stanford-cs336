import torch
from transformers import PreTrainedTokenizerBase

def compute_per_instance_dpo_loss(
    lm: torch.nn.Module,
    lm_ref: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    beta: float,
    prompt: str,
    response_chosen: str,
    response_rejected: str,
) -> torch.Tensor:
    """
    Given two language models (`lm`, and the "reference model" `lm_ref`),
    their tokenizer, the DPO beta hyperparameter, a prompt and a pair
    of responses to the prompt, computes the value of the DPO loss for this example.

    lm: torch.nn.Module
        Language model being trained.
    lm_ref: torch.nn.Module
        Reference language model.
    tokenizer: PreTrainedTokenizerBase
        Tokenizer for both language models.
    beta: float
        DPO beta hyperparameter.
    prompt: str
        Prompt for this instance of preference pair.
    response_chosen: str
        Preferred response to the prompt.
    response_rejected: str
        Rejected response to the prompt.

    Returns:
        torch.Tensor with the DPO loss for this example.
    """
    prompt_tokens = tokenizer(prompt, return_tensors="pt").input_ids
    chosen_tokens = tokenizer(prompt + response_chosen, return_tensors="pt").input_ids
    rejected_tokens = tokenizer(prompt + response_rejected, return_tensors="pt").input_ids
    chosen_logps = compute_logps(lm, chosen_tokens, prompt_tokens.shape[1])
    rejected_logps = compute_logps(lm, rejected_tokens, prompt_tokens.shape[1])
    chosen_ref_logps = compute_logps(lm_ref, chosen_tokens, prompt_tokens.shape[1])
    rejected_ref_logps = compute_logps(lm_ref, rejected_tokens, prompt_tokens.shape[1])
    chosen_logratios = chosen_logps - chosen_ref_logps
    rejected_logratios = rejected_logps - rejected_ref_logps
    loss = -torch.nn.functional.logsigmoid(beta * (chosen_logratios - rejected_logratios))
    
    return loss

def compute_logps(model, input_ids, prompt_length):
    """Helper function to compute log probabilities of a sequence"""
    with torch.no_grad():
        outputs = model(input_ids)
        logits: torch.Tensor = outputs.logits[:, prompt_length-1:-1, :]  # Shift to align with targets
        targets: torch.Tensor = input_ids[:, prompt_length:]
        log_probs = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
        token_log_probs = torch.take_along_dim(log_probs, targets.unsqueeze(-1), dim=-1)
        return token_log_probs.sum()