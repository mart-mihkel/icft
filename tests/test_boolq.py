"""Tests for BoolQ dataset loading and tokenization."""

from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from instruct.datasets.boolq import load_boolq

if TYPE_CHECKING:
    from datasets.dataset_dict import DatasetDict
    from transformers import PreTrainedTokenizerFast


def test_boolq_seqcls(
    bert_tokenizer: PreTrainedTokenizerFast,
    boolq: DatasetDict,
) -> None:
    with patch("instruct.datasets.boolq.load_dataset", return_value=boolq):
        data, _ = load_boolq(bert_tokenizer, "encoder", 0)

    assert len(data["train"]) > 0
    assert len(data["dev"]) > 0
    assert len(data["test"]) > 0

    train_sample = data["train"][0]

    assert "label" in train_sample
    assert "input_ids" in train_sample
    assert "attention_mask" in train_sample
    assert train_sample["label"] in {0, 1}


def test_boolq_causal(
    gpt2_tokenizer: PreTrainedTokenizerFast,
    boolq: DatasetDict,
) -> None:
    with patch("instruct.datasets.boolq.load_dataset", return_value=boolq):
        data, _ = load_boolq(gpt2_tokenizer, "decoder", 0)

    assert len(data["train"]) > 0
    assert len(data["dev"]) > 0
    assert len(data["test"]) > 0

    train_sample = data["train"][0]
    labels = train_sample["labels"]
    prompt_len = len(train_sample["input_ids"])

    assert "labels" in train_sample
    assert "input_ids" in train_sample
    assert "attention_mask" in train_sample
    assert len(labels) == prompt_len


def test_boolq_seq2seq(
    t5_tokenizer: PreTrainedTokenizerFast,
    boolq: DatasetDict,
) -> None:
    with patch("instruct.datasets.boolq.load_dataset", return_value=boolq):
        data, _ = load_boolq(t5_tokenizer, "encoder-decoder", 0)

    assert len(data["train"]) > 0
    assert len(data["dev"]) > 0
    assert len(data["test"]) > 0

    train_sample = data["train"][0]

    assert "labels" in train_sample
    assert "input_ids" in train_sample
    assert "attention_mask" in train_sample
    assert all(label >= 0 for label in train_sample["labels"])


def test_boolq_n_shot(
    gpt2_tokenizer: PreTrainedTokenizerFast,
    boolq: DatasetDict,
) -> None:
    n_shot = 3
    with patch("instruct.datasets.boolq.load_dataset", return_value=boolq):
        data, _ = load_boolq(gpt2_tokenizer, "encoder", n_shot)

    sample = gpt2_tokenizer.decode(data["train"][0]["input_ids"])
    assert sample.count("Passage:") == n_shot
    assert sample.count("Question:") == n_shot
    assert sample.count("Answer:") == n_shot


def test_boolq_invalid_n_shot(
    bert_tokenizer: PreTrainedTokenizerFast,
    boolq: DatasetDict,
) -> None:
    with (
        pytest.raises(ValueError, match="requested more examples than exist"),
        patch("instruct.datasets.boolq.load_dataset", return_value=boolq),
    ):
        load_boolq(bert_tokenizer, "encoder", 100)
