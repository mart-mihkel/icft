"""Dataset name dispatch: load, subsample, and fetch system prompts by name."""

from typing import TYPE_CHECKING

from saspbft.datasets.boolq import boolq_sys_prompt, load_boolq
from saspbft.datasets.estner import estner_sys_prompt, load_estner
from saspbft.datasets.multinerd import load_multinerd, multinerd_sys_prompt
from saspbft.datasets.obl import load_obl, obl_sys_prompt
from saspbft.datasets.wic import load_wic, wic_sys_prompt
from saspbft.logging import logger

if TYPE_CHECKING:
    from collections.abc import Callable

    from datasets.dataset_dict import DatasetDict
    from datasets.splits import Split
    from transformers import PreTrainedTokenizerFast

    from saspbft.types import Architecture, DatasetInfo, DatasetLoader, DatasetName

type _SysPromptFn = Callable[[PreTrainedTokenizerFast, Architecture], str]

DATASET_LOADERS: dict[DatasetName, DatasetLoader] = {
    "boolq": load_boolq,
    "wic": load_wic,
    "estner": load_estner,
    "multinerd": load_multinerd,
    "obl": load_obl,
}

SYS_PROMPT_FNS: dict[DatasetName, _SysPromptFn] = {
    "boolq": boolq_sys_prompt,
    "wic": wic_sys_prompt,
    "estner": estner_sys_prompt,
    "multinerd": multinerd_sys_prompt,
    "obl": obl_sys_prompt,
}


def get_sys_prompt(
    dataset: DatasetName,
    tokenizer: PreTrainedTokenizerFast,
    arch: Architecture,
) -> str:
    """Return a dataset's system prompt without loading or tokenizing its data."""
    return SYS_PROMPT_FNS[dataset](tokenizer, arch)


def load_data(
    tokenizer: PreTrainedTokenizerFast,
    dataset: DatasetName,
    arch: Architecture,
    n_shot: int,
    *,
    n_train_samples: int | None = None,
    n_val_samples: int | None = None,
    split: Split | None = None,
    n_virtual: int = 0,
) -> tuple[DatasetDict, DatasetInfo]:
    """Load the named dataset and optionally subsample its train/validation splits."""
    logger.info("load '%s' dataset", dataset)
    data, info = DATASET_LOADERS[dataset](
        tokenizer,
        arch,
        n_shot=n_shot,
        n_virtual=n_virtual,
        split=split,
    )

    for subsplit in data:
        n_truncated = sum(data[subsplit]["truncated"])
        if n_truncated:
            logger.warning(
                "may have truncated %d/%d %s examples to fit the max sequence length ",
                n_truncated,
                len(data[subsplit]),
                subsplit,
            )

    data = data.remove_columns("truncated")

    if n_train_samples is not None:
        n_train = len(data["train"])
        if n_train_samples > n_train:
            n_train_samples = n_train
            logger.warning("requested more train samples than in dataset %d", n_train)

        if n_train_samples < n_train:
            data["train"] = data["train"].select(range(n_train_samples))
            logger.warning("using %d of %d train samples", n_train_samples, n_train)

    if n_val_samples is not None:
        n_val = len(data["validation"])
        if n_val_samples > n_val:
            n_val_samples = n_val
            logger.warning(
                "requested more validation samples than in dataset %d",
                n_val,
            )

        if n_val_samples < n_val:
            data["validation"] = data["validation"].select(range(n_val_samples))
            logger.warning(
                "using %d of %d validation samples",
                n_val_samples,
                n_val,
            )

    return data, info
