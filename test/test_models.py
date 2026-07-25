"""End-to-end forward pass tests across model architectures and datasets."""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast
from unittest.mock import patch

import pytest
from peft import PeftModel
from torch.nn import Linear
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    BertForSequenceClassification,
)

from saspbft.datasets.boolq import load_boolq
from saspbft.datasets.estner import load_estner
from saspbft.datasets.multinerd import load_multinerd
from saspbft.datasets.obl import load_obl
from saspbft.datasets.wic import load_wic
from saspbft.modeling.collate import get_collator
from saspbft.modeling.tuning import get_pt_model
from saspbft.types import DatasetInfo

if TYPE_CHECKING:
    from datasets.dataset_dict import DatasetDict
    from peft import PromptTuningConfig
    from transformers import PreTrainedModel, PreTrainedTokenizerFast

    from saspbft.types import Architecture

type Loader = Callable[..., tuple[DatasetDict, DatasetInfo]]

_INFO = DatasetInfo(
    id2label={0: "0", 1: "1"},
    label2id={"0": 0, "1": 1},
    system_prompt="test",
)


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


@pytest.fixture(scope="session")
def bert(bert_path: str) -> BertForSequenceClassification:
    return BertForSequenceClassification.from_pretrained(bert_path)


@pytest.fixture(scope="session")
def pt_bert(bert_path: str, bert_tokenizer: PreTrainedTokenizerFast) -> PeftModel:
    return get_pt_model("random", bert_tokenizer, bert_path, "encoder", _INFO)


@pytest.fixture(scope="session")
def gpt2(gpt2_path: str) -> PreTrainedModel:
    model = AutoModelForCausalLM.from_pretrained(gpt2_path)
    return cast("PreTrainedModel", model)


@pytest.fixture(scope="session")
def pt_gpt2(gpt2_path: str, gpt2_tokenizer: PreTrainedTokenizerFast) -> PeftModel:
    return get_pt_model("random", gpt2_tokenizer, gpt2_path, "decoder", _INFO)


@pytest.fixture(scope="session")
def t5(t5_path: str) -> PreTrainedModel:
    model = AutoModelForSeq2SeqLM.from_pretrained(t5_path)
    return cast("PreTrainedModel", model)


@pytest.fixture(scope="session")
def pt_t5(t5_path: str, t5_tokenizer: PreTrainedTokenizerFast) -> PeftModel:
    return get_pt_model("random", t5_tokenizer, t5_path, "encoder-decoder", _INFO)


@pytest.fixture(scope="session")
def llama(llama_path: str) -> PreTrainedModel:
    model = AutoModelForCausalLM.from_pretrained(llama_path)
    return cast("PreTrainedModel", model)


@pytest.fixture(scope="session")
def pt_llama(llama_path: str, llama_tokenizer: PreTrainedTokenizerFast) -> PeftModel:
    return get_pt_model("random", llama_tokenizer, llama_path, "decoder", _INFO)


@pytest.fixture(scope="session")
def gemma(gemma_path: str) -> PreTrainedModel:
    model = AutoModelForCausalLM.from_pretrained(gemma_path)
    return cast("PreTrainedModel", model)


@pytest.fixture(scope="session")
def pt_gemma(gemma_path: str, gemma_tokenizer: PreTrainedTokenizerFast) -> PeftModel:
    return get_pt_model("random", gemma_tokenizer, gemma_path, "decoder", _INFO)


@pytest.fixture(scope="session")
def qwen(qwen_path: str) -> PreTrainedModel:
    model = AutoModelForCausalLM.from_pretrained(qwen_path)
    return cast("PreTrainedModel", model)


@pytest.fixture(scope="session")
def pt_qwen(qwen_path: str, qwen_tokenizer: PreTrainedTokenizerFast) -> PeftModel:
    return get_pt_model("random", qwen_tokenizer, qwen_path, "decoder", _INFO)


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
    n_virtual: int = 0,
) -> tuple[DatasetDict, DatasetInfo]:
    kwargs = {
        **spec.extra_kwargs,
        "n_shot": 0,
        "n_virtual": n_virtual,
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
    n_virtual: int = 0,
) -> None:
    data, info = _load_dataset(
        dataset_spec,
        tokenizer,
        arch,
        request,
        n_virtual,
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
        n_virtual=ptcfg.num_virtual_tokens,
    )
