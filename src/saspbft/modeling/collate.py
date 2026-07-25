"""Batch collation: pad tokenized features to a shared, multiple-of-8 length."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor
from transformers import DataCollator, DataCollatorWithPadding

from saspbft.constants import PAD_MULTIPLE, SENTINEL_TOKEN
from saspbft.logging import logger

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerFast

    from saspbft.types import Architecture


@dataclass
class Collator:
    """Pad a batch of encoded features to a shared, multiple-of-8 length."""

    tokenizer: PreTrainedTokenizerFast
    arch: Architecture

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Tensor]:
        """Pad and stack a batch of tokenized features into tensors."""
        pad = self.tokenizer.pad_token_id
        mul = PAD_MULTIPLE
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

            labels.append(_labels + [SENTINEL_TOKEN] * (max_labels - len(_labels)))
            inputs.append(_inputs + [pad] * (max_len - len(_inputs)))
            attn.append(_attn + [0] * (max_len - len(_attn)))
            tti.append((_tti or [0] * len(_inputs)) + [0] * (max_len - len(_inputs)))

        return {
            "labels": torch.tensor(labels),
            "input_ids": torch.tensor(inputs),
            "token_type_ids": torch.tensor(tti),
            "attention_mask": torch.tensor(attn),
        }


def get_collator(
    tokenizer: PreTrainedTokenizerFast,
    arch: Architecture,
) -> DataCollator:
    """Build the padding collator appropriate for the given architecture."""
    logger.debug("init data collator for '%s'", arch)
    if arch == "encoder":
        return DataCollatorWithPadding(
            tokenizer=tokenizer,
            pad_to_multiple_of=PAD_MULTIPLE,
        )

    return Collator(tokenizer=tokenizer, arch=arch)
