"""Tests for MultiNERD dataset loading and tokenization."""

from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from instruct.datasets.multinerd import _join_spans, load_multinerd

if TYPE_CHECKING:
    from datasets.dataset_dict import DatasetDict
    from transformers import PreTrainedTokenizerFast


def test_join_spans() -> None:
    tokens = ["Kuulus", "kohver", "Eston", "Kohver"]
    tag_ids = [0, 0, 1, 2]
    jtokens, jtags = _join_spans(tokens=tokens, tag_ids=tag_ids)

    assert jtokens == ["Kuulus", "kohver", "Eston Kohver"]
    assert jtags == [-1, -1, 0]


def test_multinerd_seqcls(
    bert_tokenizer: PreTrainedTokenizerFast,
    multinerd: DatasetDict,
) -> None:
    with patch("instruct.datasets.multinerd.load_dataset", return_value=multinerd):
        data, _ = load_multinerd(bert_tokenizer, "encoder", 0, False)

    assert len(data["train"]) > 0
    assert len(data["dev"]) > 0
    assert len(data["test"]) > 0

    train_sample = data["train"][0]

    assert "labels" in train_sample
    assert "input_ids" in train_sample
    assert "attention_mask" in train_sample
    assert isinstance(train_sample["labels"], int)


def test_multinerd_causal(
    gpt2_tokenizer: PreTrainedTokenizerFast,
    multinerd: DatasetDict,
) -> None:
    with patch("instruct.datasets.multinerd.load_dataset", return_value=multinerd):
        data, _ = load_multinerd(gpt2_tokenizer, "decoder", 0, False)

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


def test_multinerd_seq2seq(
    t5_tokenizer: PreTrainedTokenizerFast,
    multinerd: DatasetDict,
) -> None:
    with patch("instruct.datasets.multinerd.load_dataset", return_value=multinerd):
        data, _ = load_multinerd(t5_tokenizer, "encoder-decoder", 0, False)

    assert len(data["train"]) > 0
    assert len(data["dev"]) > 0
    assert len(data["test"]) > 0

    train_sample = data["train"][0]

    assert "labels" in train_sample
    assert "input_ids" in train_sample
    assert "attention_mask" in train_sample
    assert all(label >= 0 for label in train_sample["labels"])


def test_multinerd_n_shot(
    gpt2_tokenizer: PreTrainedTokenizerFast,
    multinerd: DatasetDict,
) -> None:
    n_shot = 3
    with patch("instruct.datasets.multinerd.load_dataset", return_value=multinerd):
        data, _ = load_multinerd(gpt2_tokenizer, "encoder", n_shot, False)

    sample = gpt2_tokenizer.decode(data["train"][0]["input_ids"])
    assert sample.count("Sentence:") == n_shot
    assert sample.count("Entity:") == n_shot
    assert sample.count("Tag:") == n_shot


def test_multinerd_invalid_n_shot(
    bert_tokenizer: PreTrainedTokenizerFast,
    multinerd: DatasetDict,
) -> None:
    with (
        pytest.raises(ValueError, match="requested more examples than exist"),
        patch("datasets.load.load_dataset", return_value=multinerd),
    ):
        load_multinerd(bert_tokenizer, "encoder", 100, False)
