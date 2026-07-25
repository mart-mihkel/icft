"""Tests for architecture family inference."""

from transformers import BertConfig, GPT2Config, T5Config

from saspbft.modeling.arch import get_arch


def test_get_arch_override_short_circuits() -> None:
    assert get_arch(T5Config(), override="decoder") == "decoder"


def test_get_arch_encoder_decoder() -> None:
    assert get_arch(T5Config()) == "encoder-decoder"


def test_get_arch_encoder() -> None:
    assert get_arch(BertConfig()) == "encoder"


def test_get_arch_decoder() -> None:
    assert get_arch(GPT2Config()) == "decoder"
