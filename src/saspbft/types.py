"""Shared type aliases and typed dicts for datasets and models."""

from typing import TYPE_CHECKING, Literal, Protocol, TypedDict

if TYPE_CHECKING:
    from datasets.dataset_dict import DatasetDict
    from datasets.splits import Split
    from transformers import EvalPrediction, PreTrainedTokenizerFast

type LogLevel = Literal["debug", "info", "warning", "error"]
type Architecture = Literal["encoder", "decoder", "encoder-decoder"]
type DatasetName = Literal["multinerd", "estner", "boolq", "wic", "obl"]
type PrefixInit = Literal["pretrained", "random"]


class DatasetInfo(TypedDict):
    """Label mappings and system prompt needed to load and prompt a dataset."""

    id2label: dict[int, str]
    label2id: dict[str, int]
    system_prompt: str


class DatasetLoader(Protocol):
    """Common interface implemented by every `load_<dataset>` function."""

    def __call__(
        self,
        tokenizer: PreTrainedTokenizerFast,
        arch: Architecture,
        *,
        n_shot: int = 0,
        n_virtual: int = 0,
        split: Split | None = None,
    ) -> tuple[DatasetDict, DatasetInfo]:
        """Load, tokenize, and prompt-format a dataset."""
        ...


class ComputeMetricsFn(Protocol):
    """Common interface implemented by every `compute_metrics_*` function."""

    def __call__(
        self,
        eval_pred: EvalPrediction,
        *,
        compute_result: bool = True,
    ) -> dict[str, int | float]:
        """Accumulate a batch of predictions and compute metrics once done."""
        ...
