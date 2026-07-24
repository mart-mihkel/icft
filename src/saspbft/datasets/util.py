"""Shared dataset loading and collation utilities."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import torch
from torch import Tensor
from transformers import (
    AutoTokenizer,
    DataCollator,
    DataCollatorWithPadding,
    PreTrainedTokenizerFast,
)

from saspbft.constants import IGNORE_TOKEN
from saspbft.datasets.boolq import load_boolq
from saspbft.datasets.estner import load_estner
from saspbft.datasets.multinerd import load_multinerd
from saspbft.datasets.obl import load_obl
from saspbft.datasets.wic import load_wic
from saspbft.logging import logger

if TYPE_CHECKING:
    from datasets.dataset_dict import DatasetDict
    from datasets.splits import Split

    from saspbft.types import Architecture, DatasetInfo, DatasetLoader, DatasetName

DATASET_LOADERS: dict[DatasetName, DatasetLoader] = {
    "boolq": load_boolq,
    "wic": load_wic,
    "estner": load_estner,
    "multinerd": load_multinerd,
    "obl": load_obl,
}


@dataclass
class Collator:
    """Pad a batch of encoded features to a shared, multiple-of-8 length."""

    tokenizer: PreTrainedTokenizerFast
    arch: Architecture

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Tensor]:
        """Pad and stack a batch of tokenized features into tensors."""
        pad = self.tokenizer.pad_token_id
        mul = 8
        max_len = max(len(feature["input_ids"]) for feature in features)
        max_len = (max_len + mul - 1) // mul * mul

        if self.arch == "encoder-decoder":
            max_labels = max(len(feature["labels"]) for feature in features)
            max_labels = (max_labels + mul - 1) // mul * mul
        else:
            max_labels = max_len

        labels = []
        inputs = []
        attn = []
        tti = []

        for feature in features:
            _labels = feature.get("labels", [])
            _inputs = feature["input_ids"]
            _attn = feature["attention_mask"]
            _tti = feature.get("token_type_ids")

            labels.append(_labels + [IGNORE_TOKEN] * (max_labels - len(_labels)))
            inputs.append(_inputs + [pad] * (max_len - len(_inputs)))
            attn.append(_attn + [0] * (max_len - len(_attn)))
            tti.append((_tti or [0] * len(_inputs)) + [0] * (max_len - len(_inputs)))

        return {
            "labels": torch.tensor(labels),
            "input_ids": torch.tensor(inputs),
            "token_type_ids": torch.tensor(tti),
            "attention_mask": torch.tensor(attn),
        }


def load_data(
    tokenizer: PreTrainedTokenizerFast,
    dataset: DatasetName,
    arch: Architecture,
    n_shot: int,
    *,
    n_train_samples: int | None = None,
    n_val_samples: int | None = None,
    split: Split | None = None,
) -> tuple[DatasetDict, DatasetInfo]:
    """Load the named dataset and optionally subsample its train/validation splits."""
    logger.info("load '%s' dataset", dataset)
    data, info = DATASET_LOADERS[dataset](tokenizer, arch, n_shot, split)

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


def load_tokenizer(model_path: str) -> PreTrainedTokenizerFast:
    """Load a tokenizer, falling back to the eos token for padding if needed."""
    logger.info("load pretrained tokenizer for '%s'", model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer = cast("PreTrainedTokenizerFast", tokenizer)

    if tokenizer.pad_token is None:
        logger.warning("tokenizer doesn't have a padding token, using eos")
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return tokenizer


def get_collator(
    tokenizer: PreTrainedTokenizerFast,
    arch: Architecture,
) -> DataCollator:
    """Build the padding collator appropriate for the given architecture."""
    logger.debug("init data collator for '%s'", arch)
    if arch == "encoder":
        return DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8)

    return Collator(tokenizer=tokenizer, arch=arch)
