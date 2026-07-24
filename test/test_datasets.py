"""Tests for dataset loading, prompt-formatting, and tokenization."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from instruct.constants import IGNORE_TOKEN
from instruct.datasets.boolq import load_boolq
from instruct.datasets.estner import _join_spans as join_spans_estner
from instruct.datasets.estner import label2id as estner_label2id
from instruct.datasets.estner import load_estner
from instruct.datasets.multinerd import _join_spans as join_spans_multinerd
from instruct.datasets.multinerd import load_multinerd
from instruct.datasets.obl import load_obl
from instruct.datasets.util import get_collator
from instruct.datasets.wic import load_wic

if TYPE_CHECKING:
    from datasets.dataset_dict import DatasetDict
    from transformers import PreTrainedTokenizerFast

    from instruct.types import Architecture, DatasetInfo

type Loader = Callable[..., tuple[DatasetDict, DatasetInfo]]
type LabelKey = str


def _is_estner_label(label: object) -> bool:
    return isinstance(label, int) and label in estner_label2id.values()


def _is_multinerd_label(label: object) -> bool:
    return isinstance(label, int)


def _check_estner_causal_labels(labels: list[int]) -> None:
    first_unmasked = next(
        (i for i, label in enumerate(labels) if label != IGNORE_TOKEN),
        -1,
    )

    assert first_unmasked > 0


@dataclass(frozen=True)
class DatasetSpec:
    """Everything needed to drive the generic loader tests for one dataset."""

    name: str
    loader: Loader
    patch_target: str | None
    fixture: str | None
    label_key: LabelKey
    is_valid_label: Callable[[object], bool]
    n_shot_markers: tuple[str, ...]
    invalid_n_shot: int
    invalid_n_shot_match: str
    extra_args: tuple[bool, ...] = ()
    check_causal_labels: Callable[[list[int]], None] | None = None


DATASET_SPECS = [
    DatasetSpec(
        name="boolq",
        loader=load_boolq,
        patch_target="instruct.datasets.boolq.load_dataset",
        fixture="boolq",
        label_key="label",
        is_valid_label=lambda label: label in {0, 1},
        n_shot_markers=("Passage:", "Question:", "Answer:"),
        invalid_n_shot=100,
        invalid_n_shot_match="requested more examples than exist",
    ),
    DatasetSpec(
        name="wic",
        loader=load_wic,
        patch_target="instruct.datasets.wic.load_dataset",
        fixture="wic",
        label_key="label",
        is_valid_label=lambda label: label in {0, 1},
        n_shot_markers=("Sentence 1:", "Sentence 2:", "Word:", "Answer (yes/no):"),
        invalid_n_shot=100,
        invalid_n_shot_match="requested more examples than exist",
    ),
    DatasetSpec(
        name="estner",
        loader=load_estner,
        patch_target="instruct.datasets.estner.load_dataset",
        fixture="estner",
        label_key="labels",
        is_valid_label=_is_estner_label,
        n_shot_markers=("Lause:", "Nimeüksus:", "Märgend:"),
        invalid_n_shot=100,
        invalid_n_shot_match="requested more examples than exist",
        check_causal_labels=_check_estner_causal_labels,
    ),
    DatasetSpec(
        name="multinerd",
        loader=load_multinerd,
        patch_target="instruct.datasets.multinerd.load_dataset",
        fixture="multinerd",
        label_key="labels",
        is_valid_label=_is_multinerd_label,
        n_shot_markers=("Sentence:", "Entity:", "Tag:"),
        invalid_n_shot=100,
        invalid_n_shot_match="requested more examples than exist",
        extra_args=(False,),
    ),
    DatasetSpec(
        name="obl",
        loader=load_obl,
        patch_target=None,
        fixture=None,
        label_key="label",
        is_valid_label=lambda label: label in {0, 1, 2, 3, 4},
        n_shot_markers=("Lause:", "Fraas:", "Kategooria:"),
        invalid_n_shot=100_000,
        invalid_n_shot_match="requested more than 1616 examples",
    ),
]


@pytest.fixture(params=DATASET_SPECS, ids=[spec.name for spec in DATASET_SPECS])
def dataset_spec(request: pytest.FixtureRequest) -> DatasetSpec:
    return request.param


def _load(
    spec: DatasetSpec,
    tokenizer: PreTrainedTokenizerFast,
    arch: Architecture,
    n_shot: int,
    request: pytest.FixtureRequest,
) -> tuple[DatasetDict, DatasetInfo]:
    if spec.patch_target is None or spec.fixture is None:
        return spec.loader(tokenizer, arch, n_shot, *spec.extra_args)

    dataset = request.getfixturevalue(spec.fixture)
    with patch(spec.patch_target, return_value=dataset):
        return spec.loader(tokenizer, arch, n_shot, *spec.extra_args)


def test_seqcls(
    dataset_spec: DatasetSpec,
    bert_tokenizer: PreTrainedTokenizerFast,
    request: pytest.FixtureRequest,
) -> None:
    data, _ = _load(dataset_spec, bert_tokenizer, "encoder", 0, request)

    assert len(data["train"]) > 0
    assert len(data["dev"]) > 0
    assert len(data["test"]) > 0

    train_sample = data["train"][0]

    assert dataset_spec.label_key in train_sample
    assert "input_ids" in train_sample
    assert "attention_mask" in train_sample
    assert dataset_spec.is_valid_label(train_sample[dataset_spec.label_key])


def test_causal(
    dataset_spec: DatasetSpec,
    gpt2_tokenizer: PreTrainedTokenizerFast,
    request: pytest.FixtureRequest,
) -> None:
    data, _ = _load(dataset_spec, gpt2_tokenizer, "decoder", 0, request)

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

    if dataset_spec.check_causal_labels is not None:
        dataset_spec.check_causal_labels(labels)


def test_seq2seq(
    dataset_spec: DatasetSpec,
    t5_tokenizer: PreTrainedTokenizerFast,
    request: pytest.FixtureRequest,
) -> None:
    data, _ = _load(dataset_spec, t5_tokenizer, "encoder-decoder", 0, request)

    assert len(data["train"]) > 0
    assert len(data["dev"]) > 0
    assert len(data["test"]) > 0

    train_sample = data["train"][0]

    assert "labels" in train_sample
    assert "input_ids" in train_sample
    assert "attention_mask" in train_sample
    assert all(label >= 0 for label in train_sample["labels"])


def test_n_shot(
    dataset_spec: DatasetSpec,
    gpt2_tokenizer: PreTrainedTokenizerFast,
    request: pytest.FixtureRequest,
) -> None:
    n_shot = 3
    data, _ = _load(dataset_spec, gpt2_tokenizer, "encoder", n_shot, request)

    sample = gpt2_tokenizer.decode(data["train"][0]["input_ids"])
    for marker in dataset_spec.n_shot_markers:
        assert sample.count(marker) == n_shot


def test_invalid_n_shot(
    dataset_spec: DatasetSpec,
    bert_tokenizer: PreTrainedTokenizerFast,
    request: pytest.FixtureRequest,
) -> None:
    with pytest.raises(ValueError, match=dataset_spec.invalid_n_shot_match):
        _load(
            dataset_spec,
            bert_tokenizer,
            "encoder",
            dataset_spec.invalid_n_shot,
            request,
        )


def test_collator_with_labels(gpt2_tokenizer: PreTrainedTokenizerFast) -> None:
    features = [
        {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1], "labels": [10, 20]},
        {"input_ids": [4, 5], "attention_mask": [1, 1], "labels": [30]},
    ]

    collator = get_collator(gpt2_tokenizer, "decoder")
    batch = collator(features)

    assert "input_ids" in batch
    assert "attention_mask" in batch
    assert "labels" in batch

    assert batch["input_ids"].shape[1] % 8 == 0
    assert batch["attention_mask"].shape[1] % 8 == 0
    assert batch["labels"].shape[1] % 8 == 0

    assert batch["input_ids"][0][-1] == gpt2_tokenizer.eos_token_id
    assert batch["attention_mask"][0][-1] == 0
    assert batch["labels"][0][-1] == IGNORE_TOKEN


def test_collator_with_no_labels(gpt2_tokenizer: PreTrainedTokenizerFast) -> None:
    features = [
        {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]},
        {"input_ids": [4, 5], "attention_mask": [1, 1]},
    ]

    collator = get_collator(gpt2_tokenizer, "decoder")
    batch = collator(features)

    assert "input_ids" in batch
    assert "attention_mask" in batch
    assert "labels" in batch

    assert batch["input_ids"].shape[1] % 8 == 0
    assert batch["attention_mask"].shape[1] % 8 == 0
    assert batch["labels"].shape[1] % 8 == 0

    assert batch["input_ids"][0][-1] == gpt2_tokenizer.eos_token_id
    assert batch["attention_mask"][0][-1] == 0
    assert batch["labels"][0][-1] == IGNORE_TOKEN


def test_join_spans_estner() -> None:
    tokens = ["Kuulus", "kohver", "Eston", "Kohver"]
    tags = ["O", "O", "B-PER", "I-PER"]
    jtokens, jtags = join_spans_estner(tokens=tokens, tags=tags)

    assert jtokens == ["Kuulus", "kohver", "Eston Kohver"]
    assert jtags == ["O", "O", "PER"]


def test_join_spans_multinerd() -> None:
    tokens = ["Kuulus", "kohver", "Eston", "Kohver"]
    tag_ids = [0, 0, 1, 2]
    jtokens, jtags = join_spans_multinerd(tokens=tokens, tag_ids=tag_ids)

    assert jtokens == ["Kuulus", "kohver", "Eston Kohver"]
    assert jtags == [-1, -1, 0]
