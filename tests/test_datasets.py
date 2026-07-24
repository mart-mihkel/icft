"""Tests for the shared data collator."""

from typing import TYPE_CHECKING

from instruct.constants import ignore_token
from instruct.datasets.util import get_collator

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerFast


def test_collator_with_labels(gpt2_tokenizer: PreTrainedTokenizerFast) -> None:
    features = [
        {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1], "labels": [10, 20]},
        {"input_ids": [4, 5], "attention_mask": [1, 1], "labels": [30]},
    ]

    collator = get_collator(gpt2_tokenizer, "decoder")
    batch = collator(features)

    assert "input_ids" in batch
    assert "attention_mask" in batch
    assert "labels" in batch

    assert batch["input_ids"].shape[1] % 8 == 0
    assert batch["attention_mask"].shape[1] % 8 == 0
    assert batch["labels"].shape[1] % 8 == 0

    assert batch["input_ids"][0][-1] == gpt2_tokenizer.eos_token_id
    assert batch["attention_mask"][0][-1] == 0
    assert batch["labels"][0][-1] == ignore_token


def test_collator_with_no_labels(gpt2_tokenizer: PreTrainedTokenizerFast) -> None:
    features = [
        {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]},
        {"input_ids": [4, 5], "attention_mask": [1, 1]},
    ]

    collator = get_collator(gpt2_tokenizer, "decoder")
    batch = collator(features)

    assert "input_ids" in batch
    assert "attention_mask" in batch
    assert "labels" in batch

    assert batch["input_ids"].shape[1] % 8 == 0
    assert batch["attention_mask"].shape[1] % 8 == 0
    assert batch["labels"].shape[1] % 8 == 0

    assert batch["input_ids"][0][-1] == gpt2_tokenizer.eos_token_id
    assert batch["attention_mask"][0][-1] == 0
    assert batch["labels"][0][-1] == ignore_token
