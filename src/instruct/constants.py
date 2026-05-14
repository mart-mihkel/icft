from pathlib import Path

logdir = Path("log")

encoder_model_types = frozenset(
    (
        "bert",
        "distilbert",
        "roberta",
        "deberta-v2",
        "eurobert",
        "modernbert",
    )
)

decoder_model_types = frozenset(
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

encoder_decoder_model_types = frozenset(
    (
        "t5",
        "t5gemma",
        "t5gemma2",
    )
)
