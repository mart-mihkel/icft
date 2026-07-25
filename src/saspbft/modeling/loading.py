"""Pretrained model loading, with flash-attention fallback and param freezing."""

from typing import TYPE_CHECKING

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
)

from saspbft.logging import logger

if TYPE_CHECKING:
    from collections.abc import Callable

    from torch.nn import Module
    from transformers import PreTrainedModel, PreTrainedTokenizerFast

    from saspbft.types import Architecture, DatasetInfo


def _load_with_attn_fallback[T](
    loader: Callable[..., T],
    model_path: str,
    attn: str | None,
    **kwargs: object,
) -> T:
    """Load a pretrained model, retrying without flash attention if unsupported."""
    try:
        return loader(model_path, attn_implementation=attn, **kwargs)
    except ValueError as e:
        if attn != "flash_attention_2" or "flash attention" not in str(e).lower():
            raise

        logger.warning(
            "'%s' doesn't support flash attention 2, using default",
            model_path,
        )

        return loader(model_path, attn_implementation=None, **kwargs)


def get_model(
    tokenizer: PreTrainedTokenizerFast,
    model_path: str,
    data_info: DatasetInfo,
    arch: Architecture,
    head_only: bool,
) -> PreTrainedModel:
    """Load a pretrained model for the architecture, optionally freezing the base."""
    if torch.cuda.is_available():
        dtype = torch.bfloat16
        attn = "flash_attention_2"
        logger.debug("using BF16 and flash attention 2")
    else:
        dtype = torch.float32
        attn = None
        logger.debug("using full precision and default attention")

    skip_freeze = None
    if arch == "encoder":
        logger.debug("load '%s' for sequence classification", model_path)
        model, loading_info = _load_with_attn_fallback(
            AutoModelForSequenceClassification.from_pretrained,
            model_path,
            attn,
            output_loading_info=True,
            num_labels=len(data_info["id2label"]),
            id2label=data_info["id2label"],
            label2id=data_info["label2id"],
            device_map="auto",
            dtype=dtype,
        )

        skip_freeze = loading_info["missing_keys"]
    elif arch == "decoder":
        logger.debug("load '%s' for causal language modeling", model_path)
        model = _load_with_attn_fallback(
            AutoModelForCausalLM.from_pretrained,
            model_path,
            attn,
            device_map="auto",
            dtype=dtype,
        )
    elif arch == "encoder-decoder":
        logger.debug("load '%s' for sequence to sequence", model_path)
        model = _load_with_attn_fallback(
            AutoModelForSeq2SeqLM.from_pretrained,
            model_path,
            attn,
            device_map="auto",
            dtype=dtype,
        )

    config = model.config
    if "text_config" in config:
        logger.debug("using text config of multimodal model")
        config = config.text_config

    if config.pad_token_id is None:
        logger.warning("model doesn't have a padding token, using eos")
        config.pad_token_id = tokenizer.eos_token_id

    if head_only:
        freeze(model, skip_freeze)

    return model


def freeze(model: Module, skip: set[str] | None = None) -> None:
    """Freeze all model parameters except those named in `skip`."""
    if skip is None:
        skip = set()

    logger.info("freeze base model")
    for name, param in model.named_parameters():
        if name in skip:
            logger.info("skip '%s'", name)
            continue

        param.requires_grad = False
