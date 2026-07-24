"""End-to-end forward pass tests for Gemma across supported datasets."""

from typing import TYPE_CHECKING
from unittest.mock import patch

from instruct.datasets.boolq import load_boolq
from instruct.datasets.estner import load_estner
from instruct.datasets.multinerd import load_multinerd
from instruct.datasets.obl import load_obl
from instruct.datasets.util import get_collator
from instruct.datasets.wic import load_wic

if TYPE_CHECKING:
    from datasets.dataset_dict import DatasetDict
    from peft import PeftModel
    from transformers import PreTrainedModel, PreTrainedTokenizerFast

_arch = "decoder"


def test_gemma_wic_forward(
    gemma: PreTrainedModel,
    gemma_tokenizer: PreTrainedTokenizerFast,
    wic: DatasetDict,
) -> None:
    with patch("instruct.datasets.wic.load_dataset", return_value=wic):
        data, _ = load_wic(gemma_tokenizer, _arch, 0)

    examples = [data["train"][i] for i in range(4)]
    collator = get_collator(gemma_tokenizer, _arch)

    batch = collator(examples)
    out = gemma(**batch)

    assert out.loss is not None
    assert out.logits is not None


def test_gemma_boolq_forward(
    gemma: PreTrainedModel,
    gemma_tokenizer: PreTrainedTokenizerFast,
    boolq: DatasetDict,
) -> None:
    with patch("instruct.datasets.boolq.load_dataset", return_value=boolq):
        data, _ = load_boolq(gemma_tokenizer, _arch, 0)

    examples = [data["train"][i] for i in range(4)]
    collator = get_collator(gemma_tokenizer, _arch)

    batch = collator(examples)
    out = gemma(**batch)

    assert out.loss is not None
    assert out.logits is not None


def test_gemma_estner_forward(
    gemma: PreTrainedModel,
    gemma_tokenizer: PreTrainedTokenizerFast,
    estner: DatasetDict,
) -> None:
    with patch("instruct.datasets.estner.load_dataset", return_value=estner):
        data, _ = load_estner(gemma_tokenizer, _arch, 0)

    examples = [data["train"][i] for i in range(4)]
    collator = get_collator(gemma_tokenizer, _arch)

    batch = collator(examples)
    out = gemma(**batch)

    assert out.loss is not None
    assert out.logits is not None


def test_gemma_multinerd_forward(
    gemma: PreTrainedModel,
    gemma_tokenizer: PreTrainedTokenizerFast,
    multinerd: DatasetDict,
) -> None:
    with patch("instruct.datasets.multinerd.load_dataset", return_value=multinerd):
        data, _ = load_multinerd(gemma_tokenizer, _arch, 0, False)

    examples = [data["train"][i] for i in range(4)]
    collator = get_collator(gemma_tokenizer, _arch)

    batch = collator(examples)
    out = gemma(**batch)

    assert out.loss is not None
    assert out.logits is not None


def test_gemma_obl_forward(
    gemma: PreTrainedModel,
    gemma_tokenizer: PreTrainedTokenizerFast,
) -> None:
    data, _ = load_obl(gemma_tokenizer, _arch, 0)

    examples = [data["train"][i] for i in range(4)]
    collator = get_collator(gemma_tokenizer, _arch)

    batch = collator(examples)
    out = gemma(**batch)

    assert out.loss is not None
    assert out.logits is not None


def test_pt_gemma_obl_forward(
    pt_gemma: PeftModel,
    gemma_tokenizer: PreTrainedTokenizerFast,
) -> None:
    data, _ = load_obl(gemma_tokenizer, _arch, 0)

    examples = [data["train"][i] for i in range(4)]
    collator = get_collator(gemma_tokenizer, _arch)

    batch = collator(examples)
    out = pt_gemma(**batch)

    assert out.loss is not None
    assert out.logits is not None


def test_pt_gemma_wic_forward(
    pt_gemma: PeftModel,
    gemma_tokenizer: PreTrainedTokenizerFast,
    wic: DatasetDict,
) -> None:
    with patch("instruct.datasets.wic.load_dataset", return_value=wic):
        data, _ = load_wic(gemma_tokenizer, _arch, 0)

    examples = [data["train"][i] for i in range(4)]
    collator = get_collator(gemma_tokenizer, _arch)

    batch = collator(examples)
    out = pt_gemma(**batch)

    assert out.loss is not None
    assert out.logits is not None


def test_pt_gemma_boolq_forward(
    pt_gemma: PeftModel,
    gemma_tokenizer: PreTrainedTokenizerFast,
    boolq: DatasetDict,
) -> None:
    with patch("instruct.datasets.boolq.load_dataset", return_value=boolq):
        data, _ = load_boolq(gemma_tokenizer, _arch, 0)

    examples = [data["train"][i] for i in range(4)]
    collator = get_collator(gemma_tokenizer, _arch)

    batch = collator(examples)
    out = pt_gemma(**batch)

    assert out.loss is not None
    assert out.logits is not None


def test_pt_gemma_estner_forward(
    pt_gemma: PeftModel,
    gemma_tokenizer: PreTrainedTokenizerFast,
    estner: DatasetDict,
) -> None:
    with patch("instruct.datasets.estner.load_dataset", return_value=estner):
        data, _ = load_estner(gemma_tokenizer, _arch, 0)

    examples = [data["train"][i] for i in range(4)]
    collator = get_collator(gemma_tokenizer, _arch)

    batch = collator(examples)
    out = pt_gemma(**batch)

    assert out.loss is not None
    assert out.logits is not None


def test_pt_gemma_multinerd_forward(
    pt_gemma: PeftModel,
    gemma_tokenizer: PreTrainedTokenizerFast,
    multinerd: DatasetDict,
) -> None:
    with patch("instruct.datasets.multinerd.load_dataset", return_value=multinerd):
        data, _ = load_multinerd(gemma_tokenizer, _arch, 0, False)

    examples = [data["train"][i] for i in range(4)]
    collator = get_collator(gemma_tokenizer, _arch)

    batch = collator(examples)
    out = pt_gemma(**batch)

    assert out.loss is not None
    assert out.logits is not None
