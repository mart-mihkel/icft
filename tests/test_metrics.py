"""Tests for streaming metric computation."""

import numpy as np
from transformers import EvalPrediction, PreTrainedTokenizerFast

from instruct.metrics import (
    _filter_gibberish,
    compute_metrics_causal_lm,
    compute_metrics_seq2seq,
    compute_metrics_seq_cls,
    get_metrics_fn,
)


def test_filter_gibberis() -> None:
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
        ]
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
        ]
    )

    labels = np.array([[0, 1], [0, 1]])
    eval_pred = EvalPrediction(logits, labels)

    metrics = compute_metrics_causal_lm(eval_pred, tokenizer=gpt2_tokenizer)

    assert "accuracy" in metrics


def test_init_metrics_fn(bert_tokenizer: PreTrainedTokenizerFast) -> None:
    metrics_fn = get_metrics_fn(bert_tokenizer, "encoder")
    assert metrics_fn == compute_metrics_seq_cls
