"""End-to-end forward pass tests for Llama across supported datasets."""

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


def test_llama_wic_forward(
    llama: PreTrainedModel,
    llama_tokenizer: PreTrainedTokenizerFast,
    wic: DatasetDict,
) -> None:
    with patch("instruct.datasets.wic.load_dataset", return_value=wic):
        data, _ = load_wic(llama_tokenizer, _arch, 0)

    examples = [data["train"][i] for i in range(4)]
    collator = get_collator(llama_tokenizer, _arch)

    batch = collator(examples)
    out = llama(**batch)

    assert out.loss is not None
    assert out.logits is not None


def test_llama_boolq_forward(
    llama: PreTrainedModel,
    llama_tokenizer: PreTrainedTokenizerFast,
    boolq: DatasetDict,
) -> None:
    with patch("instruct.datasets.boolq.load_dataset", return_value=boolq):
        data, _ = load_boolq(llama_tokenizer, _arch, 0)

    examples = [data["train"][i] for i in range(4)]
    collator = get_collator(llama_tokenizer, _arch)

    batch = collator(examples)
    out = llama(**batch)

    assert out.loss is not None
    assert out.logits is not None


def test_llama_estner_forward(
    llama: PreTrainedModel,
    llama_tokenizer: PreTrainedTokenizerFast,
    estner: DatasetDict,
) -> None:
    with patch("instruct.datasets.estner.load_dataset", return_value=estner):
        data, _ = load_estner(llama_tokenizer, _arch, 0)

    examples = [data["train"][i] for i in range(4)]
    collator = get_collator(llama_tokenizer, _arch)

    batch = collator(examples)
    out = llama(**batch)

    assert out.loss is not None
    assert out.logits is not None


def test_llama_multinerd_forward(
    llama: PreTrainedModel,
    llama_tokenizer: PreTrainedTokenizerFast,
    multinerd: DatasetDict,
) -> None:
    with patch("instruct.datasets.multinerd.load_dataset", return_value=multinerd):
        data, _ = load_multinerd(llama_tokenizer, _arch, 0, False)

    examples = [data["train"][i] for i in range(4)]
    collator = get_collator(llama_tokenizer, _arch)

    batch = collator(examples)
    out = llama(**batch)

    assert out.loss is not None
    assert out.logits is not None


def test_llama_obl_forward(
    llama: PreTrainedModel,
    llama_tokenizer: PreTrainedTokenizerFast,
) -> None:
    data, _ = load_obl(llama_tokenizer, _arch, 0)

    examples = [data["train"][i] for i in range(4)]
    collator = get_collator(llama_tokenizer, _arch)

    batch = collator(examples)
    out = llama(**batch)

    assert out.loss is not None
    assert out.logits is not None


def test_pt_llama_wic_forward(
    pt_llama: PeftModel,
    llama_tokenizer: PreTrainedTokenizerFast,
    wic: DatasetDict,
) -> None:
    with patch("instruct.datasets.wic.load_dataset", return_value=wic):
        data, _ = load_wic(llama_tokenizer, _arch, 0)

    examples = [data["train"][i] for i in range(4)]
    collator = get_collator(llama_tokenizer, _arch)

    batch = collator(examples)
    out = pt_llama(**batch)

    assert out.loss is not None
    assert out.logits is not None


def test_pt_llama_boolq_forward(
    pt_llama: PeftModel,
    llama_tokenizer: PreTrainedTokenizerFast,
    boolq: DatasetDict,
) -> None:
    with patch("instruct.datasets.boolq.load_dataset", return_value=boolq):
        data, _ = load_boolq(llama_tokenizer, _arch, 0)

    examples = [data["train"][i] for i in range(4)]
    collator = get_collator(llama_tokenizer, _arch)

    batch = collator(examples)
    out = pt_llama(**batch)

    assert out.loss is not None
    assert out.logits is not None


def test_pt_llama_estner_forward(
    pt_llama: PeftModel,
    llama_tokenizer: PreTrainedTokenizerFast,
    estner: DatasetDict,
) -> None:
    with patch("instruct.datasets.estner.load_dataset", return_value=estner):
        data, _ = load_estner(llama_tokenizer, _arch, 0)

    examples = [data["train"][i] for i in range(4)]
    collator = get_collator(llama_tokenizer, _arch)

    batch = collator(examples)
    out = pt_llama(**batch)

    assert out.loss is not None
    assert out.logits is not None


def test_pt_llama_multinerd_forward(
    pt_llama: PeftModel,
    llama_tokenizer: PreTrainedTokenizerFast,
    multinerd: DatasetDict,
) -> None:
    with patch("instruct.datasets.multinerd.load_dataset", return_value=multinerd):
        data, _ = load_multinerd(llama_tokenizer, _arch, 0, False)

    examples = [data["train"][i] for i in range(4)]
    collator = get_collator(llama_tokenizer, _arch)

    batch = collator(examples)
    out = pt_llama(**batch)

    assert out.loss is not None
    assert out.logits is not None


def test_pt_llama_obl_forward(
    pt_llama: PeftModel,
    llama_tokenizer: PreTrainedTokenizerFast,
) -> None:
    data, _ = load_obl(llama_tokenizer, _arch, 0)

    examples = [data["train"][i] for i in range(4)]
    collator = get_collator(llama_tokenizer, _arch)

    batch = collator(examples)
    out = pt_llama(**batch)

    assert out.loss is not None
    assert out.logits is not None
