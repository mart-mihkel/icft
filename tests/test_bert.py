"""BERT tests."""

from typing import TYPE_CHECKING, cast
from unittest import skip
from unittest.mock import patch

from torch.nn import Linear

from instruct.datasets.boolq import load_boolq
from instruct.datasets.estner import load_estner
from instruct.datasets.multinerd import load_multinerd
from instruct.datasets.obl import load_obl
from instruct.datasets.util import get_collator
from instruct.datasets.wic import load_wic

if TYPE_CHECKING:
    from datasets.dataset_dict import DatasetDict
    from peft import PeftModel
    from transformers import (
        BertForSequenceClassification,
        PreTrainedTokenizerFast,
    )

_arch = "encoder"


def test_bert_wic_forward(
    bert: BertForSequenceClassification,
    bert_tokenizer: PreTrainedTokenizerFast,
    wic: DatasetDict,
) -> None:
    with patch("instruct.datasets.wic.load_dataset", return_value=wic):
        data, _ = load_wic(bert_tokenizer, _arch, 0)

    examples = [data["train"][i] for i in range(4)]
    collator = get_collator(bert_tokenizer, _arch)

    batch = collator(examples)
    out = bert(**batch)

    assert out.loss is not None
    assert out.logits is not None


def test_bert_boolq_forward(
    bert: BertForSequenceClassification,
    bert_tokenizer: PreTrainedTokenizerFast,
    boolq: DatasetDict,
) -> None:
    with patch("instruct.datasets.boolq.load_dataset", return_value=boolq):
        data, _ = load_boolq(bert_tokenizer, _arch, 0)

    examples = [data["train"][i] for i in range(4)]
    collator = get_collator(bert_tokenizer, _arch)

    batch = collator(examples)
    out = bert(**batch)

    assert out.loss is not None
    assert out.logits is not None


def test_bert_estner_forward(
    bert: BertForSequenceClassification,
    bert_tokenizer: PreTrainedTokenizerFast,
    estner: DatasetDict,
) -> None:
    with patch("instruct.datasets.estner.load_dataset", return_value=estner):
        data, info = load_estner(bert_tokenizer, _arch, 0)

    num_labels = len(info["id2label"])
    bert.num_labels = num_labels
    bert.classifier = Linear(bert.config.hidden_size, num_labels)

    examples = [data["train"][i] for i in range(4)]
    collator = get_collator(bert_tokenizer, _arch)

    batch = collator(examples)
    out = bert(**batch)

    assert out.loss is not None
    assert out.logits is not None


def test_bert_multinerd_forward(
    bert: BertForSequenceClassification,
    bert_tokenizer: PreTrainedTokenizerFast,
    multinerd: DatasetDict,
) -> None:
    with patch("instruct.datasets.multinerd.load_dataset", return_value=multinerd):
        data, info = load_multinerd(bert_tokenizer, _arch, 0, False)

    num_labels = len(info["id2label"])
    bert.num_labels = num_labels
    bert.classifier = Linear(bert.config.hidden_size, num_labels)

    examples = [data["train"][i] for i in range(4)]
    collator = get_collator(bert_tokenizer, _arch)

    batch = collator(examples)
    out = bert(**batch)

    assert out.loss is not None
    assert out.logits is not None


def test_bert_obl_forward(
    bert: BertForSequenceClassification,
    bert_tokenizer: PreTrainedTokenizerFast,
) -> None:
    data, info = load_obl(bert_tokenizer, _arch, 0)

    num_labels = len(info["id2label"])
    bert.num_labels = num_labels
    bert.classifier = Linear(bert.config.hidden_size, num_labels)

    examples = [data["train"][i] for i in range(4)]
    collator = get_collator(bert_tokenizer, _arch)

    batch = collator(examples)
    out = bert(**batch)

    assert out.loss is not None
    assert out.logits is not None


def test_pt_bert_wic_forward(
    pt_bert: PeftModel,
    bert_tokenizer: PreTrainedTokenizerFast,
    wic: DatasetDict,
) -> None:
    with patch("instruct.datasets.wic.load_dataset", return_value=wic):
        data, _ = load_wic(bert_tokenizer, _arch, 0)

    examples = [data["train"][i] for i in range(4)]
    collator = get_collator(bert_tokenizer, _arch)

    batch = collator(examples)
    out = pt_bert(**batch)

    assert out.loss is not None
    assert out.logits is not None


@skip("sequence truncation based on number of virtual tokens in not implemented")
def test_pt_bert_boolq_forward(
    pt_bert: PeftModel,
    bert_tokenizer: PreTrainedTokenizerFast,
    boolq: DatasetDict,
) -> None:
    with patch("instruct.datasets.boolq.load_dataset", return_value=boolq):
        data, _ = load_boolq(bert_tokenizer, _arch, 0)

    examples = [data["train"][i] for i in range(4)]
    collator = get_collator(bert_tokenizer, _arch)

    batch = collator(examples)
    out = pt_bert(**batch)

    assert out.loss is not None
    assert out.logits is not None


def test_pt_bert_estner_forward(
    pt_bert: PeftModel,
    bert_tokenizer: PreTrainedTokenizerFast,
    estner: DatasetDict,
) -> None:
    with patch("instruct.datasets.estner.load_dataset", return_value=estner):
        data, info = load_estner(bert_tokenizer, _arch, 0)

    num_labels = len(info["id2label"])
    bert = cast("BertForSequenceClassification", pt_bert.base_model)
    bert.num_labels = num_labels
    bert.classifier = Linear(pt_bert.config.hidden_size, num_labels)

    examples = [data["train"][i] for i in range(4)]
    collator = get_collator(bert_tokenizer, _arch)

    batch = collator(examples)
    out = pt_bert(**batch)

    assert out.loss is not None
    assert out.logits is not None


def test_pt_bert_multinerd_forward(
    pt_bert: PeftModel,
    bert_tokenizer: PreTrainedTokenizerFast,
    multinerd: DatasetDict,
) -> None:
    with patch("instruct.datasets.multinerd.load_dataset", return_value=multinerd):
        data, info = load_multinerd(bert_tokenizer, _arch, 0, False)

    num_labels = len(info["id2label"])
    bert = cast("BertForSequenceClassification", pt_bert.base_model)
    bert.num_labels = num_labels
    bert.classifier = Linear(pt_bert.config.hidden_size, num_labels)

    examples = [data["train"][i] for i in range(4)]
    collator = get_collator(bert_tokenizer, _arch)

    batch = collator(examples)
    out = pt_bert(**batch)

    assert out.loss is not None
    assert out.logits is not None


def test_pt_bert_obl_forward(
    pt_bert: PeftModel,
    bert_tokenizer: PreTrainedTokenizerFast,
) -> None:
    data, info = load_obl(bert_tokenizer, _arch, 0)

    num_labels = len(info["id2label"])
    bert = cast("BertForSequenceClassification", pt_bert.base_model)
    bert.num_labels = num_labels
    bert.classifier = Linear(pt_bert.config.hidden_size, num_labels)

    examples = [data["train"][i] for i in range(4)]
    collator = get_collator(bert_tokenizer, _arch)

    batch = collator(examples)
    out = pt_bert(**batch)

    assert out.loss is not None
    assert out.logits is not None
