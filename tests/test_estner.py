"""Tests for EstNER dataset loading and tokenization."""

from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from instruct.constants import ignore_token
from instruct.datasets.estner import _join_spans, label2id, load_estner

if TYPE_CHECKING:
    from datasets.dataset_dict import DatasetDict
    from transformers import PreTrainedTokenizerFast


def test_join_spans() -> None:
    tokens = ["Kuulus", "kohver", "Eston", "Kohver"]
    tags = ["O", "O", "B-PER", "I-PER"]
    jtokens, jtags = _join_spans(tokens=tokens, tags=tags)

    assert jtokens == ["Kuulus", "kohver", "Eston Kohver"]
    assert jtags == ["O", "O", "PER"]


def test_estner_seqcls(
    bert_tokenizer: PreTrainedTokenizerFast,
    estner: DatasetDict,
) -> None:
    with patch("instruct.datasets.estner.load_dataset", return_value=estner):
        data, _ = load_estner(bert_tokenizer, "encoder", 0)

    assert len(data["train"]) > 0
    assert len(data["dev"]) > 0
    assert len(data["test"]) > 0

    train_sample = data["train"][0]

    assert "labels" in train_sample
    assert "input_ids" in train_sample
    assert "attention_mask" in train_sample
    assert isinstance(train_sample["labels"], int)
    assert train_sample["labels"] in label2id.values()


def test_estner_causal(
    gpt2_tokenizer: PreTrainedTokenizerFast,
    estner: DatasetDict,
) -> None:
    with patch("instruct.datasets.estner.load_dataset", return_value=estner):
        data, _ = load_estner(gpt2_tokenizer, "decoder", 0)

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

    first_non_masked = next(
        (i for i, label in enumerate(labels) if label != ignore_token), -1
    )

    assert first_non_masked > 0


def test_estner_seq2seq(
    t5_tokenizer: PreTrainedTokenizerFast,
    estner: DatasetDict,
) -> None:
    with patch("instruct.datasets.estner.load_dataset", return_value=estner):
        data, _ = load_estner(t5_tokenizer, "encoder-decoder", 0)

    assert len(data["train"]) > 0
    assert len(data["dev"]) > 0
    assert len(data["test"]) > 0

    train_sample = data["train"][0]

    assert "labels" in train_sample
    assert "input_ids" in train_sample
    assert "attention_mask" in train_sample
    assert all(label >= 0 for label in train_sample["labels"])


def test_estner_n_shot(
    gpt2_tokenizer: PreTrainedTokenizerFast,
    estner: DatasetDict,
) -> None:
    n_shot = 3
    with patch("instruct.datasets.estner.load_dataset", return_value=estner):
        data, _ = load_estner(gpt2_tokenizer, "encoder", n_shot)

    sample = gpt2_tokenizer.decode(data["train"][0]["input_ids"])
    assert sample.count("Lause:") == n_shot
    assert sample.count("Nimeüksus:") == n_shot
    assert sample.count("Märgend:") == n_shot


def test_estner_invalid_n_shot(
    bert_tokenizer: PreTrainedTokenizerFast,
    estner: DatasetDict,
) -> None:
    with (
        pytest.raises(ValueError, match="requested more examples than exist"),
        patch("instruct.datasets.estner.load_dataset", return_value=estner),
    ):
        load_estner(bert_tokenizer, "encoder", 100)
