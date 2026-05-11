import pytest
from transformers import PreTrainedTokenizerFast

from instruct.datasets.obl import load_obl


def test_obl_seqcls(bert_tokenizer: PreTrainedTokenizerFast) -> None:
    data, _ = load_obl(bert_tokenizer, "encoder", 0)

    assert len(data["train"]) > 0
    assert len(data["dev"]) > 0
    assert len(data["test"]) > 0

    train_sample = data["train"][0]

    assert "label" in train_sample
    assert "input_ids" in train_sample
    assert "attention_mask" in train_sample
    assert train_sample["label"] in {0, 1, 2, 3, 4}


def test_obl_causal(gpt2_tokenizer: PreTrainedTokenizerFast) -> None:
    data, _ = load_obl(gpt2_tokenizer, "decoder", 0)

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


def test_obl_seq2seq(t5_tokenizer: PreTrainedTokenizerFast) -> None:
    data, _ = load_obl(t5_tokenizer, "encoder-decoder", 0)

    assert len(data["train"]) > 0
    assert len(data["dev"]) > 0
    assert len(data["test"]) > 0

    train_sample = data["train"][0]

    assert "labels" in train_sample
    assert "input_ids" in train_sample
    assert "attention_mask" in train_sample
    assert all(label >= 0 for label in train_sample["labels"])


def test_obl_n_shot(gpt2_tokenizer: PreTrainedTokenizerFast) -> None:
    n_shot = 3
    data, _ = load_obl(gpt2_tokenizer, "encoder", n_shot)

    sample = gpt2_tokenizer.decode(data["train"][0]["input_ids"])
    assert sample.count("Lause:") == n_shot
    assert sample.count("Fraas:") == n_shot
    assert sample.count("Kategooria:") == n_shot


def test_obl_invalid_n_shot(
    bert_tokenizer: PreTrainedTokenizerFast,
) -> None:
    with pytest.raises(ValueError, match="requested more than 1616 examples"):
        load_obl(bert_tokenizer, "encoder", 100000)
