from transformers import PreTrainedTokenizerBase
from torch import Tensor
import torch

def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizerBase,
) -> dict[str, Tensor]:
    """Tokenize the prompt and output strings, and construct a mask that is 1
    for the response tokens and 0 for other tokens (prompt or padding).

    Args:
        prompt_strs: list[str], the prompt strings.
        output_strs: list[str], the output strings.
        tokenizer: PreTrainedTokenizer, the tokenizer to use.

    Returns:
        dict[str, torch.Tensor]:
            "input_ids": torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
                the tokenized prompt and output strings, with the final token sliced off.
            "labels": torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
                shifted input_ids (i.e., the input_ids without the first token).
            "response_mask": torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
                a mask on the response tokens in `labels`.
    """
    batch_size = len(prompt_strs)
    prompts = tokenizer.batch_encode_plus(prompt_strs, add_special_tokens=False, return_length=True)
    outputs = tokenizer.batch_encode_plus(output_strs, add_special_tokens=False)
    encoded = {
        "input_ids": [prompt + output for prompt, output in zip(prompts["input_ids"], outputs["input_ids"])], 
        "attention_mask": [prompt + output for prompt, output in zip(prompts["attention_mask"], outputs["attention_mask"])], 
    }
    encoded = tokenizer.pad(encoded_inputs=encoded, return_tensors="pt")
    encoded_ids = encoded["input_ids"]
    max_prompt_and_output_len = encoded_ids.size(1)
    idx = torch.arange(max_prompt_and_output_len).expand(batch_size, max_prompt_and_output_len)
    prompt_lengths = torch.tensor(prompts["length"])[:, None]
    ret: dict[str, Tensor] = {
        "input_ids": encoded_ids[..., :-1], 
        "labels": encoded_ids[..., 1:].clone(), 
        "response_mask": (torch.where(prompt_lengths <= idx, True, False) & encoded["attention_mask"])[..., 1:]
    }
    return ret