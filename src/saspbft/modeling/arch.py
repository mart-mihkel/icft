"""Architecture family inference from a model config."""

from typing import TYPE_CHECKING

from transformers import AutoModelForMaskedLM, AutoModelForSeq2SeqLM

from saspbft.logging import logger

if TYPE_CHECKING:
    from transformers import PreTrainedConfig

    from saspbft.types import Architecture


def get_arch(
    config: PreTrainedConfig,
    override: Architecture | None = None,
) -> Architecture:
    """Infer the model architecture family from its config."""
    if override is not None:
        logger.debug("using overridden architecture '%s'", override)
        return override

    cls = type(config)
    if config.is_encoder_decoder or cls in AutoModelForSeq2SeqLM._model_mapping:
        arch: Architecture = "encoder-decoder"
    elif cls in AutoModelForMaskedLM._model_mapping:
        arch = "encoder"
    else:
        arch = "decoder"

    logger.debug("inferred model architecture '%s' for '%s'", arch, config.model_type)
    return arch
