"""Tests for streaming metric computation."""

import numpy as np
import pytest
from transformers import EvalPrediction, PreTrainedTokenizerFast

from saspbft.modeling.metrics import (
    _compute_bleu,
    _compute_perplexity,
    _compute_rouge,
    _filter_gibberish,
    compute_metrics_causal_lm,
    compute_metrics_seq2seq,
    compute_metrics_seq_cls,
    get_metrics_fn,
)


def test_filter_gibberish() -> None:
    ref = [" yes", " no", " yes"]
    pred = [" dog", " no", " yes"]
    assert _filter_gibberish(ref, pred) == ["<gibberish>", " no", " yes"]


def test_seq_cls() -> None:
    logits = np.array([[2.0, 1.0, 0.0], [2.0, 1.0, 0.0]])
    labels = np.array([0, 0])
    eval_pred = EvalPrediction(logits, labels)
    metrics = compute_metrics_seq_cls(eval_pred)

    assert metrics["accuracy"] == 1.0


def test_seq2seq(t5_tokenizer: PreTrainedTokenizerFast) -> None:
    logits = np.array(
        [
            [[1.0, 4.0, 3.0, 2.0, 5.0], [1.0, 4.0, 3.0, 2.0, 5.0]],
            [[1.0, 4.0, 3.0, 2.0, 5.0], [1.0, 4.0, 3.0, 2.0, 5.0]],
        ],
    )
    preds = np.argmax(logits, axis=-1)

    labels = np.array([[4, 4], [4, 4]])
    eval_pred = EvalPrediction(preds, labels)
    metrics = compute_metrics_seq2seq(eval_pred, tokenizer=t5_tokenizer)

    assert metrics["accuracy"] == 1.0


def test_causal_lm(gpt2_tokenizer: PreTrainedTokenizerFast) -> None:
    logits = np.array(
        [
            [[5.0, 4.0, 3.0, 2.0, 1.0], [5.0, 4.0, 3.0, 2.0, 1.0]],
            [[5.0, 4.0, 3.0, 2.0, 1.0], [5.0, 4.0, 3.0, 2.0, 1.0]],
        ],
    )

    labels = np.array([[0, 1], [0, 1]])
    eval_pred = EvalPrediction(logits, labels)

    metrics = compute_metrics_causal_lm(eval_pred, tokenizer=gpt2_tokenizer)

    assert "accuracy" in metrics


def test_init_metrics_fn(bert_tokenizer: PreTrainedTokenizerFast) -> None:
    metrics_fn = get_metrics_fn(bert_tokenizer, "encoder")
    assert metrics_fn == compute_metrics_seq_cls


def test_seq_cls_accumulate_only_returns_empty() -> None:
    logits = np.array([[2.0, 1.0, 0.0]])
    labels = np.array([0])
    eval_pred = EvalPrediction(logits, labels)

    accumulated = compute_metrics_seq_cls(eval_pred, compute_result=False)
    assert accumulated == {}

    metrics = compute_metrics_seq_cls(eval_pred, compute_result=True)
    assert metrics["accuracy"] == 1.0


def test_seq2seq_accumulate_only_returns_empty(
    t5_tokenizer: PreTrainedTokenizerFast,
) -> None:
    preds = np.array([[4, 4]])
    labels = np.array([[4, 4]])
    eval_pred = EvalPrediction(preds, labels)

    accumulated = compute_metrics_seq2seq(
        eval_pred,
        tokenizer=t5_tokenizer,
        compute_result=False,
    )

    assert accumulated == {}

    metrics = compute_metrics_seq2seq(eval_pred, tokenizer=t5_tokenizer)
    assert metrics["accuracy"] == 1.0


def test_causal_lm_accumulate_only_returns_empty(
    gpt2_tokenizer: PreTrainedTokenizerFast,
) -> None:
    logits = np.array([[[5.0, 4.0, 3.0, 2.0, 1.0], [5.0, 4.0, 3.0, 2.0, 1.0]]])
    labels = np.array([[0, 1]])
    eval_pred = EvalPrediction(logits, labels)

    accumulated = compute_metrics_causal_lm(
        eval_pred,
        tokenizer=gpt2_tokenizer,
        compute_result=False,
    )

    assert accumulated == {}

    metrics = compute_metrics_causal_lm(eval_pred, tokenizer=gpt2_tokenizer)
    assert "accuracy" in metrics


def test_causal_lm_trims_virtual_prefix_tokens(
    gpt2_tokenizer: PreTrainedTokenizerFast,
) -> None:
    # predictions run 2 timesteps longer than labels, as happens when the
    # model's output includes positions for prompt-tuning's virtual tokens
    logits = np.zeros((1, 4, 5))
    logits[0, 2, 0] = 10.0
    logits[0, 3, 1] = 10.0

    labels = np.array([[0, 1]])
    eval_pred = EvalPrediction(logits, labels)

    metrics = compute_metrics_causal_lm(eval_pred, tokenizer=gpt2_tokenizer)

    assert "accuracy" in metrics


def test_compute_perplexity_of_confident_correct_predictions() -> None:
    labels = np.array([0, 1])
    logits = np.array([[10.0, 0.0], [0.0, 10.0]])

    result = _compute_perplexity(labels, logits)

    assert result["perplexity"] == pytest.approx(1.0, abs=1e-3)


def test_compute_bleu_happy_path() -> None:
    sentence = "the cat sat on the warm windowsill"
    result = _compute_bleu([sentence], [sentence])
    assert result["bleu"] == pytest.approx(1.0)


def test_compute_bleu_returns_empty_in_child_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("saspbft.modeling.metrics._bleu.compute", lambda **_: None)
    assert _compute_bleu(["a"], ["a"]) == {}


def test_compute_rouge_happy_path() -> None:
    result = _compute_rouge(["the cat sat"], ["the cat sat"])
    assert result["rouge1"] == pytest.approx(1.0)


def test_compute_rouge_returns_empty_in_child_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("saspbft.modeling.metrics._rouge.compute", lambda **_: None)
    assert _compute_rouge(["a"], ["a"]) == {}
