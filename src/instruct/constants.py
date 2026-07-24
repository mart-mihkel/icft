"""Shared constants for model architectures and dataset labels."""

from pathlib import Path

IGNORE_TOKEN = -100
LOGDIR = Path("log")

ENC_TYPES = frozenset(
    (
        "bert",
        "distilbert",
        "roberta",
        "deberta-v2",
        "eurobert",
        "modernbert",
    )
)

DEC_TYPES = frozenset(
    (
        "gpt2",
        "gpt_neox",
        "gemma",
        "gemma2",
        "gemma3",
        "gemma3_text",
        "gemma4",
        "gemma4_text",
        "qwen2",
        "qwen2_5",
        "qwen3",
        "qwen3_5",
        "qwen3_5_text",
        "llama",
        "llama4",
    )
)

ENCDEC_TYPES = frozenset(
    (
        "t5",
        "t5gemma",
        "t5gemma2",
    )
)
