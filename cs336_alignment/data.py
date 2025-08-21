import os
import json
import torch
import random
from transformers import PreTrainedTokenizerBase
from typing import Any, List

prompt_template_path = "./cs336_alignment/prompts/alpaca_sft.prompt"

class SFTDataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        tokenizer: PreTrainedTokenizerBase,
        dataset_path: str | os.PathLike,
        seq_length: int,
        shuffle: bool = False,
    ):
        self.tokenizer = tokenizer
        with open(prompt_template_path, "r") as f:
            self.prompt_template = f.read()
        with open(dataset_path, "r") as fp:
            raw = [json.loads(line) for line in fp.readlines()]
            raw = [line for line in raw]
        self.data: list[dict[str, torch.Tensor]] = []
        tokens = []
        for item in raw:
            text = self.prompt_template.strip().format(
                instruction=item["prompt"],
                response=item["response"] + '<|end_of_text|>'
            )
            sub = tokenizer(text, add_special_tokens=True, padding=False, truncation=False)['input_ids']
            tokens.extend(sub)
        for i in range(0, len(tokens), seq_length):
            if i + seq_length >= len(tokens):
                continue
            input_ids = tokens[i : i + seq_length]
            labels = tokens[i + 1 : i + 1 + seq_length]
            self.data.append({
                "input_ids": torch.tensor(input_ids),
                "labels": torch.tensor(labels), 
            })
        
        if shuffle:
            random.shuffle(self.data)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.data[idx]

    def __iter__(self):
        return iter(self.data)


def iterate_batches(
    dataset: torch.utils.data.Dataset,
    batch_size: int,
    shuffle: bool,
) -> torch.utils.data.DataLoader:
    """
    Given a PyTorch Dataset, return an iterable over batches of size `batch_size`.
    Iterating through the returned iterable should constitute one epoch over the Dataset.

    Args:
        dataset: Dataset
            Dataset to emit batches from.
        batch_size: int
            Number of examples to include per batch.
        shuffle: bool
            If true, shuffle examples before batching them.

    Returns:
        Iterable over batches, where each batch has size `batch_size`.
    """
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)