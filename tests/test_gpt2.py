"""End-to-end forward pass tests for GPT-2 across supported datasets."""

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


def test_gpt2_wic_forward(
    gpt2: PreTrainedModel,
    gpt2_tokenizer: PreTrainedTokenizerFast,
    wic: DatasetDict,
) -> None:
    with patch("instruct.datasets.wic.load_dataset", return_value=wic):
        data, _ = load_wic(gpt2_tokenizer, _arch, 0)

    examples = [data["train"][i] for i in range(4)]
    collator = get_collator(gpt2_tokenizer, _arch)

    batch = collator(examples)
    out = gpt2(**batch)

    assert out.loss is not None
    assert out.logits is not None


def test_gpt2_boolq_forward(
    gpt2: PreTrainedModel,
    gpt2_tokenizer: PreTrainedTokenizerFast,
    boolq: DatasetDict,
) -> None:
    with patch("instruct.datasets.boolq.load_dataset", return_value=boolq):
        data, _ = load_boolq(gpt2_tokenizer, _arch, 0)

    examples = [data["train"][i] for i in range(4)]
    collator = get_collator(gpt2_tokenizer, _arch)

    batch = collator(examples)
    out = gpt2(**batch)

    assert out.loss is not None
    assert out.logits is not None


def test_gpt2_estner_forward(
    gpt2: PreTrainedModel,
    gpt2_tokenizer: PreTrainedTokenizerFast,
    estner: DatasetDict,
) -> None:
    with patch("instruct.datasets.estner.load_dataset", return_value=estner):
        data, _ = load_estner(gpt2_tokenizer, _arch, 0)

    examples = [data["train"][i] for i in range(4)]
    collator = get_collator(gpt2_tokenizer, _arch)

    batch = collator(examples)
    out = gpt2(**batch)

    assert out.loss is not None
    assert out.logits is not None


def test_gpt2_multinerd_forward(
    gpt2: PreTrainedModel,
    gpt2_tokenizer: PreTrainedTokenizerFast,
    multinerd: DatasetDict,
) -> None:
    with patch("instruct.datasets.multinerd.load_dataset", return_value=multinerd):
        data, _ = load_multinerd(gpt2_tokenizer, _arch, 0, False)

    examples = [data["train"][i] for i in range(4)]
    collator = get_collator(gpt2_tokenizer, _arch)

    batch = collator(examples)
    out = gpt2(**batch)

    assert out.loss is not None
    assert out.logits is not None


def test_gpt2_obl_forward(
    gpt2: PreTrainedModel,
    gpt2_tokenizer: PreTrainedTokenizerFast,
) -> None:
    data, _ = load_obl(gpt2_tokenizer, _arch, 0)

    examples = [data["train"][i] for i in range(4)]
    collator = get_collator(gpt2_tokenizer, _arch)

    batch = collator(examples)
    out = gpt2(**batch)

    assert out.loss is not None
    assert out.logits is not None


def test_pt_gpt2_wic_forward(
    pt_gpt2: PeftModel,
    gpt2_tokenizer: PreTrainedTokenizerFast,
    wic: DatasetDict,
) -> None:
    with patch("instruct.datasets.wic.load_dataset", return_value=wic):
        data, _ = load_wic(gpt2_tokenizer, _arch, 0)

    examples = [data["train"][i] for i in range(4)]
    collator = get_collator(gpt2_tokenizer, _arch)

    batch = collator(examples)
    out = pt_gpt2(**batch)

    assert out.loss is not None
    assert out.logits is not None


def test_pt_gpt2_boolq_forward(
    pt_gpt2: PeftModel,
    gpt2_tokenizer: PreTrainedTokenizerFast,
    boolq: DatasetDict,
) -> None:
    with patch("instruct.datasets.boolq.load_dataset", return_value=boolq):
        data, _ = load_boolq(gpt2_tokenizer, _arch, 0)

    examples = [data["train"][i] for i in range(4)]
    collator = get_collator(gpt2_tokenizer, _arch)

    batch = collator(examples)
    out = pt_gpt2(**batch)

    assert out.loss is not None
    assert out.logits is not None


def test_pt_gpt2_estner_forward(
    pt_gpt2: PeftModel,
    gpt2_tokenizer: PreTrainedTokenizerFast,
    estner: DatasetDict,
) -> None:
    with patch("instruct.datasets.estner.load_dataset", return_value=estner):
        data, _ = load_estner(gpt2_tokenizer, _arch, 0)

    examples = [data["train"][i] for i in range(4)]
    collator = get_collator(gpt2_tokenizer, _arch)

    batch = collator(examples)
    out = pt_gpt2(**batch)

    assert out.loss is not None
    assert out.logits is not None


def test_pt_gpt2_multinerd_forward(
    pt_gpt2: PeftModel,
    gpt2_tokenizer: PreTrainedTokenizerFast,
    multinerd: DatasetDict,
) -> None:
    with patch("instruct.datasets.multinerd.load_dataset", return_value=multinerd):
        data, _ = load_multinerd(gpt2_tokenizer, _arch, 0, False)

    examples = [data["train"][i] for i in range(4)]
    collator = get_collator(gpt2_tokenizer, _arch)

    batch = collator(examples)
    out = pt_gpt2(**batch)

    assert out.loss is not None
    assert out.logits is not None


def test_pt_gpt2_obl_forward(
    pt_gpt2: PeftModel,
    gpt2_tokenizer: PreTrainedTokenizerFast,
) -> None:
    data, _ = load_obl(gpt2_tokenizer, _arch, 0)

    examples = [data["train"][i] for i in range(4)]
    collator = get_collator(gpt2_tokenizer, _arch)

    batch = collator(examples)
    out = pt_gpt2(**batch)

    assert out.loss is not None
    assert out.logits is not None
