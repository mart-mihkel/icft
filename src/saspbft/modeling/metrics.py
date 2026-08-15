"""Streaming metric computation for encoder, decoder, and encoder-decoder Trainers."""

from collections import Counter
from typing import TYPE_CHECKING

import evaluate
import numpy as np
from scipy.special import log_softmax
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.fft import Tensor
from transformers import EvalPrediction  # noqa: TC002

from saspbft.constants import SENTINEL_TOKEN
from saspbft.logging import format_counts, logger

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerFast

    from saspbft.types import Architecture, ComputeMetricsFn

_bleu = evaluate.load("bleu")
_rouge = evaluate.load("rouge")

# global state during `trainer.evaluate` to collect batched eval loop outputs
_labels: list[np.ndarray] = []
_preds: list[np.ndarray] = []


def _batch_to_numpy(eval_pred: EvalPrediction) -> tuple[np.ndarray, np.ndarray]:
    batch_labels = eval_pred.label_ids
    if isinstance(batch_labels, tuple):
        batch_labels = batch_labels[0]

    if isinstance(batch_labels, Tensor):
        batch_labels = batch_labels.detach().cpu().numpy()

    batch_logits = eval_pred.predictions
    if isinstance(batch_logits, tuple):
        batch_logits = batch_logits[0]

    if isinstance(batch_logits, Tensor):
        batch_logits = batch_logits.detach().cpu().numpy()

    batch_preds = np.argmax(batch_logits, axis=-1)
    return batch_labels, batch_preds


def _filter_gibberish(references: list[str], predictions: list[str]) -> list[str]:
    refset = set(references)
    return [pred if pred in refset else "<gibberish>" for pred in predictions]


def _compute_classification_metrics(
    labels: np.ndarray | list,
    preds: np.ndarray | list,
) -> dict[str, float]:
    logger.debug("compute classification metrics")

    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average="macro", zero_division=0)
    recall = recall_score(labels, preds, average="macro", zero_division=0)
    f1 = f1_score(labels, preds, average="macro", zero_division=0)
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


def _compute_perplexity(labels: np.ndarray, logits: np.ndarray) -> dict[str, float]:
    logger.debug("compute perplexity")

    idx = np.arange(labels.shape[0])
    log_probs = log_softmax(logits, axis=-1)[idx, labels]
    perplexity = np.exp(-log_probs.mean())
    return {"perplexity": perplexity}


def _compute_bleu(
    references: list[str],
    predictions: list[str],
) -> dict[str, float]:
    logger.debug("compute BLEU")

    res = _bleu.compute(
        predictions=predictions,
        references=references,
    )  # ty: ignore[missing-argument]

    if res is None:
        logger.warning("BLEU evaluation was run in a child process")
        return {}

    return {"bleu": res["bleu"]}


def _compute_rouge(
    references: list[str],
    predictions: list[str],
) -> dict[str, float]:
    logger.debug("compute ROUGE")

    res = _rouge.compute(
        predictions=predictions,
        references=references,
        rouge_types=["rouge1", "rouge2"],
    )  # ty: ignore[missing-argument]

    if res is None:
        logger.warning("ROUGE evaluation was run in a child process")
        return {}

    return res


def compute_metrics_seq_cls(
    eval_pred: EvalPrediction,
    *,
    compute_result: bool = True,
) -> dict[str, float]:
    """Accumulate a batch of encoder predictions and compute metrics once done."""
    labels, preds = _batch_to_numpy(eval_pred)
    _labels.extend(labels)
    _preds.extend(preds)

    if not compute_result:
        return {}

    all_labels = np.array(_labels)
    all_preds = np.array(_preds)
    mask = all_labels != SENTINEL_TOKEN

    _labels.clear()
    _preds.clear()

    _labels_count = Counter(all_labels[mask].tolist())
    _preds_count = Counter(all_preds[mask].tolist())

    logger.debug("labels: %s", format_counts(_labels_count))
    logger.debug("predictions: %s", format_counts(_preds_count))

    return _compute_classification_metrics(all_labels[mask], all_preds[mask])


def compute_metrics_seq2seq(
    eval_pred: EvalPrediction,
    tokenizer: PreTrainedTokenizerFast,
    *,
    compute_result: bool = True,
) -> dict[str, float]:
    """Accumulate a batch of seq2seq predictions and compute metrics once done."""
    batch_labels = eval_pred.label_ids
    batch_preds = eval_pred.predictions

    if isinstance(batch_labels, tuple):
        batch_labels = batch_labels[0]

    if isinstance(batch_preds, tuple):
        batch_preds = batch_preds[0]

    _labels.extend(batch_labels)
    _preds.extend(batch_preds)

    if not compute_result:
        return {}

    labels = []
    preds = []
    for label, pred in zip(_labels, _preds, strict=True):
        label_mask = label != SENTINEL_TOKEN
        labels.append(label[label_mask])

        pred_mask = pred != tokenizer.pad_token_id
        preds.append(pred[pred_mask])

    _labels.clear()
    _preds.clear()

    references = tokenizer.batch_decode(labels, skip_special_tokens=True)
    predictions = tokenizer.batch_decode(preds, skip_special_tokens=True)
    predictions = _filter_gibberish(references, predictions)

    _refs_count = Counter(references)
    _preds_count = Counter(predictions)

    logger.debug("references: %s", format_counts(_refs_count))
    logger.debug("predictions: %s", format_counts(_preds_count))

    return _compute_classification_metrics(references, predictions)


def compute_metrics_causal_lm(
    eval_pred: EvalPrediction,
    tokenizer: PreTrainedTokenizerFast,
    *,
    compute_result: bool = True,
) -> dict[str, float]:
    """Accumulate a batch of causal LM predictions and compute metrics once done."""
    labels, preds = _batch_to_numpy(eval_pred)

    _, label_dim = labels.shape
    _, pred_dim = preds.shape
    if label_dim < pred_dim:
        virtual_dim = pred_dim - label_dim
        preds = preds[:, virtual_dim:]

    _labels.extend(labels)
    _preds.extend(preds)

    if not compute_result:
        return {}

    labels = []
    preds = []
    for label, pred in zip(_labels, _preds, strict=True):
        shifted_label = label[1:]
        shifted_pred = pred[:-1]

        mask = shifted_label != SENTINEL_TOKEN
        labels.append(shifted_label[mask])
        preds.append(shifted_pred[mask])

    _labels.clear()
    _preds.clear()

    references = tokenizer.batch_decode(labels, skip_special_tokens=True)
    predictions = tokenizer.batch_decode(preds, skip_special_tokens=True)
    predictions = _filter_gibberish(references, predictions)

    _refs_count = Counter(references)
    _preds_count = Counter(predictions)

    logger.debug("references: %s", format_counts(_refs_count))
    logger.debug("predictions: %s", format_counts(_preds_count))

    return _compute_classification_metrics(references, predictions)


def get_metrics_fn(
    tokenizer: PreTrainedTokenizerFast,
    arch: Architecture,
) -> ComputeMetricsFn:
    """Return the `compute_metrics` function appropriate for the architecture."""
    logger.debug("init compute metrics for '%s'", arch)
    if arch == "encoder":
        return compute_metrics_seq_cls

    if arch == "decoder":
        return lambda eval_pred, *, compute_result=True: compute_metrics_causal_lm(
            eval_pred=eval_pred,
            compute_result=compute_result,
            tokenizer=tokenizer,
        )

    if arch == "encoder-decoder":
        return lambda eval_pred, *, compute_result=True: compute_metrics_seq2seq(
            eval_pred=eval_pred,
            compute_result=compute_result,
            tokenizer=tokenizer,
        )
    return None
