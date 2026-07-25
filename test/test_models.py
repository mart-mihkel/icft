"""End-to-end forward pass tests across model architectures and datasets."""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast
from unittest.mock import patch

import pytest
from peft import PeftModel
from torch.nn import Linear

from saspbft.datasets.boolq import load_boolq
from saspbft.datasets.estner import load_estner
from saspbft.datasets.multinerd import load_multinerd
from saspbft.datasets.obl import load_obl
from saspbft.datasets.util import get_collator
from saspbft.datasets.wic import load_wic

if TYPE_CHECKING:
    from datasets.dataset_dict import DatasetDict
    from peft import PromptTuningConfig
    from transformers import (
        BertForSequenceClassification,
        PreTrainedModel,
        PreTrainedTokenizerFast,
    )

    from saspbft.types import Architecture, DatasetInfo

type Loader = Callable[..., tuple[DatasetDict, DatasetInfo]]


@dataclass(frozen=True)
class ModelSpec:
    """A model fixture name paired with the architecture it belongs to."""

    name: str
    arch: Architecture


@dataclass(frozen=True)
class DatasetSpec:
    """A dataset loader plus the fixture/patch needed to feed it fake data."""

    name: str
    loader: Loader
    patch_target: str | None
    needs_head: bool
    extra_kwargs: dict[str, bool] = field(default_factory=dict)


MODEL_SPECS = [
    pytest.param(ModelSpec("bert", "encoder"), id="bert"),
    pytest.param(ModelSpec("gpt2", "decoder"), id="gpt2"),
    pytest.param(ModelSpec("t5", "encoder-decoder"), id="t5"),
    pytest.param(ModelSpec("llama", "decoder"), id="llama", marks=pytest.mark.slow),
    pytest.param(ModelSpec("gemma", "decoder"), id="gemma", marks=pytest.mark.slow),
    pytest.param(ModelSpec("qwen", "decoder"), id="qwen", marks=pytest.mark.slow),
]

DATASET_SPECS = [
    DatasetSpec(
        "wic",
        load_wic,
        "saspbft.datasets.wic.load_dataset",
        needs_head=False,
    ),
    DatasetSpec(
        "boolq",
        load_boolq,
        "saspbft.datasets.boolq.load_dataset",
        needs_head=False,
    ),
    DatasetSpec(
        "estner",
        load_estner,
        "saspbft.datasets.estner.load_dataset",
        needs_head=True,
    ),
    DatasetSpec(
        "multinerd",
        load_multinerd,
        "saspbft.datasets.multinerd.load_dataset",
        needs_head=True,
        extra_kwargs={"filter_en": False},
    ),
    DatasetSpec(
        "obl",
        load_obl,
        None,
        needs_head=True,
    ),
]


@pytest.fixture(params=MODEL_SPECS)
def model_spec(request: pytest.FixtureRequest) -> ModelSpec:
    return request.param


@pytest.fixture(params=DATASET_SPECS, ids=[spec.name for spec in DATASET_SPECS])
def dataset_spec(request: pytest.FixtureRequest) -> DatasetSpec:
    return request.param


@pytest.fixture
def tokenizer(
    model_spec: ModelSpec,
    request: pytest.FixtureRequest,
) -> PreTrainedTokenizerFast:
    return request.getfixturevalue(f"{model_spec.name}_tokenizer")


def _resize_classifier_head(
    model: PreTrainedModel | PeftModel,
    num_labels: int,
) -> None:
    base = model.base_model if isinstance(model, PeftModel) else model
    target = cast("BertForSequenceClassification", base)
    target.num_labels = num_labels
    classifier = Linear(model.config.hidden_size, num_labels)
    target.classifier = classifier.to(device=model.device, dtype=base.dtype)


def _load_dataset(
    spec: DatasetSpec,
    tokenizer: PreTrainedTokenizerFast,
    arch: Architecture,
    request: pytest.FixtureRequest,
    num_virtual_tokens: int = 0,
) -> tuple[DatasetDict, DatasetInfo]:
    kwargs = {
        **spec.extra_kwargs,
        "n_shot": 0,
        "num_virtual_tokens": num_virtual_tokens,
    }
    if spec.patch_target is None:
        return spec.loader(tokenizer, arch, **kwargs)

    dataset = request.getfixturevalue(spec.name)
    with patch(spec.patch_target, return_value=dataset):
        return spec.loader(tokenizer, arch, **kwargs)


def _run_forward(
    model: PreTrainedModel | PeftModel,
    tokenizer: PreTrainedTokenizerFast,
    arch: Architecture,
    dataset_spec: DatasetSpec,
    request: pytest.FixtureRequest,
    *,
    num_virtual_tokens: int = 0,
) -> None:
    data, info = _load_dataset(
        dataset_spec,
        tokenizer,
        arch,
        request,
        num_virtual_tokens,
    )

    if dataset_spec.needs_head and arch == "encoder":
        _resize_classifier_head(model, len(info["id2label"]))

    examples = [data["train"][i] for i in range(4)]
    collator = get_collator(tokenizer, arch)
    batch = collator(examples)
    batch = {key: value.to(model.device) for key, value in batch.items()}
    out = model(**batch)

    assert out.loss is not None
    assert out.logits is not None


def test_forward(
    model_spec: ModelSpec,
    dataset_spec: DatasetSpec,
    tokenizer: PreTrainedTokenizerFast,
    request: pytest.FixtureRequest,
) -> None:
    model = request.getfixturevalue(model_spec.name)
    _run_forward(model, tokenizer, model_spec.arch, dataset_spec, request)


def test_pt_forward(
    model_spec: ModelSpec,
    dataset_spec: DatasetSpec,
    tokenizer: PreTrainedTokenizerFast,
    request: pytest.FixtureRequest,
) -> None:
    model = request.getfixturevalue(f"pt_{model_spec.name}")
    ptcfg = cast("PromptTuningConfig", model.peft_config["default"])
    _run_forward(
        model,
        tokenizer,
        model_spec.arch,
        dataset_spec,
        request,
        num_virtual_tokens=ptcfg.num_virtual_tokens,
    )
