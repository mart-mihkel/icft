from typing import cast

from pytest import approx
from transformers.trainer_utils import EvalPrediction

from icftner.datasets.multinerd import (
    _prepare_prompt_bert,
    compute_multinerd_prompted_metrics,
)


def test_prepare_prompt_bert():
    target = "[CLS] fish [SEP] How much is the fish ? [SEP]"
    prompt_tokens = _prepare_prompt_bert(
        target_token="fish",
        tokens=["How", "much", "is", "the", "fish", "?"],
        system_tokens=[],
    )

    assert target == " ".join(prompt_tokens)


def test_prepare_system_prompt_bert():
    target = "[CLS] Find the NER tags . [SEP] fish [SEP] How much is the fish ? [SEP]"
    prompt_tokens = _prepare_prompt_bert(
        target_token="fish",
        tokens=["How", "much", "is", "the", "fish", "?"],
        system_tokens=["Find", "the", "NER", "tags", "."],
    )

    assert target == " ".join(prompt_tokens)


def test_compute_multinerd_metrics():
    labels = [0, 0, 1, 1, 2, 2]
    logits = [
        [1, 0, 0],
        [1, 0, 0],
        [0, 0, 1],
        [0, 0, 1],
        [0, 0, 1],
        [1, 0, 0],
    ]

    eval_pred = cast(EvalPrediction, (logits, labels))
    eval_metrics = compute_multinerd_prompted_metrics(eval_pred)

    assert eval_metrics["accuracy"] == approx(0.5)
    assert eval_metrics["precision"] == approx(0.5)
    assert eval_metrics["recall"] == approx(0.5)
    assert eval_metrics["f1"] == approx(0.4)
