import math
from collections import Counter

import numpy as np
import pytest

from .adapters import run_get_batch


def test_get_batch():
    dataset = np.arange(0, 100)
    context_length = 7
    batch_size = 32
    device = "cpu"

    # Sanity check to make sure that the random samples are indeed somewhat random.
    starting_indices = Counter()
    num_iters = 1000
    for _ in range(num_iters):
        x, y = run_get_batch(
            dataset=dataset,
            batch_size=batch_size,
            context_length=context_length,
            device=device,
        )

        # Make sure the shape is correct
        assert x.shape == (batch_size, context_length)
        assert y.shape == (batch_size, context_length)

        # Make sure the y's are always offset by 1
        np.testing.assert_allclose((x + 1).detach().numpy(), y.detach().numpy())

        starting_indices.update(x[:, 0].tolist())

    # Make sure we never sample an invalid start index
    num_possible_starting_indices = len(dataset) - context_length
    assert max(starting_indices) == num_possible_starting_indices - 1
    assert min(starting_indices) == 0
    # Expected # of times that we see each starting index
    expected_count = (num_iters * batch_size) / num_possible_starting_indices
    standard_deviation = math.sqrt(
        (num_iters * batch_size) * (1 / num_possible_starting_indices) * (1 - (1 / num_possible_starting_indices))
    )
    # Range for expected outcomes (mu +/- 5sigma). For a given index,
    # this should happen 99.99994% of the time of the time.
    # So, in the case where we have 93 possible start indices,
    # the entire test should pass with 99.9944202% of the time
    occurrences_lower_bound = expected_count - 5 * standard_deviation
    occurrences_upper_bound = expected_count + 5 * standard_deviation

    for starting_index, count in starting_indices.items():
        if count < occurrences_lower_bound:
            raise ValueError(
                f"Starting index {starting_index} occurs {count} times, but expected at least {occurrences_lower_bound}"
            )
        if count > occurrences_upper_bound:
            raise ValueError(
                f"Starting index {starting_index} occurs {count} times, but expected at most {occurrences_upper_bound}"
            )

    with pytest.raises((RuntimeError, AssertionError)) as excinfo:
        # We're assuming that cuda:99 is an invalid device ordinal.
        # Just adding this here to make sure that the device flag is
        # being handled.
        run_get_batch(
            dataset=dataset,
            batch_size=batch_size,
            context_length=context_length,
            device="cuda:99",
        )
        assert "CUDA error" in str(excinfo.value) or "Torch not compiled with CUDA enabled" in str(excinfo.value)

# Assignment5
import json
import logging
import math

import torch
from transformers import AutoTokenizer

from .adapters import get_packed_sft_dataset, run_iterate_batches
from .common import FIXTURES_PATH

logger = logging.getLogger(__name__)


def test_packed_sft_dataset():
    sft_sample_path = FIXTURES_PATH / "sft_sample.jsonl"
    tokenizer = AutoTokenizer.from_pretrained(FIXTURES_PATH / "Meta-Llama-3-8B")
    seq_length = 32
    packed_sft_dataset = get_packed_sft_dataset(
        tokenizer=tokenizer,
        dataset_path=sft_sample_path,
        seq_length=seq_length,
        shuffle=False,
    )

    with open(FIXTURES_PATH / "tokenized_sft_sample.json") as f:
        expected_examples = json.load(f)

    assert len(packed_sft_dataset) == len(expected_examples)

    for i, (example, expected_example) in enumerate(zip(packed_sft_dataset, expected_examples)):
        assert example["input_ids"].tolist() == expected_example["input_ids"]
        assert example["labels"].tolist() == expected_example["labels"]

    # Check that shuffling works by ensuring that it returns different data
    # than the unshuffled dataset. Note that there's a small chance that it
    # just happens that the shuffling preserves the original order.
    shuffled_packed_sft_dataset = get_packed_sft_dataset(
        tokenizer=tokenizer,
        dataset_path=sft_sample_path,
        seq_length=seq_length,
        shuffle=True,
    )
    all_unshuffled_examples = []
    all_shuffled_examples = []
    for example, shuffled_example in zip(
        packed_sft_dataset, shuffled_packed_sft_dataset
    ):
        all_unshuffled_examples.append({k: v.tolist() for k, v in example.items()})
        all_shuffled_examples.append(
            {k: v.tolist() for k, v in shuffled_example.items()}
        )
    assert all_unshuffled_examples != all_shuffled_examples


def test_iterate_batches():
    sft_sample_path = FIXTURES_PATH / "sft_sample.jsonl"
    tokenizer = AutoTokenizer.from_pretrained(FIXTURES_PATH / "Meta-Llama-3-8B")
    seq_length = 32
    batch_size = 8
    packed_sft_dataset = get_packed_sft_dataset(
        tokenizer=tokenizer,
        dataset_path=sft_sample_path,
        seq_length=seq_length,
        shuffle=True,
    )
    train_dataloader = run_iterate_batches(
        dataset=packed_sft_dataset, batch_size=batch_size, shuffle=True
    )
    assert len(train_dataloader) == math.ceil(75 / batch_size)
    for batch_idx, batch in enumerate(train_dataloader):
        # Make sure each of input_ids and labels is a (batch_size, seq_length) tensor, except
        # for the last batch (which can be less than batch_size items)
        if batch_idx != len(train_dataloader) - 1:
            assert batch["input_ids"].shape == (batch_size, seq_length)
            assert batch["labels"].shape == (batch_size, seq_length)

        assert (
            batch["input_ids"].dtype == torch.long
            or batch["input_ids"].dtype == torch.int64
        )
        assert (
            batch["labels"].dtype == torch.long or batch["labels"].dtype == torch.int64
        )
