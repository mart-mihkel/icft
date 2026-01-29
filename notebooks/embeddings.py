import marimo

__generated_with = "0.19.6"
app = marimo.App()

with app.setup:
    from typing import cast, Annotated

    from torch import Tensor
    from transformers import (
        AutoTokenizer,
        AutoModel,
        PreTrainedTokenizer,
        ModernBertModel,
    )

    from icft.datasets.multinerd import Multinerd


@app.function
def encode_prefix(
    prefix: str,
    model: ModernBertModel,
    tokenizer: PreTrainedTokenizer,
) -> Annotated[Tensor, "prefix emb"]:
    prefix_tokenized = tokenizer(prefix, return_tensors="pt")
    token_ids = prefix_tokenized["input_ids"][0]
    return model.get_input_embeddings().forward(token_ids)


@app.function
def decode_prefix(
    prefix_embeddings: Annotated[Tensor, "prefix emb"],
    model: ModernBertModel,
    tokenizer: PreTrainedTokenizer,
) -> str | list[str]:
    voc_embeddings = cast(Tensor, model.get_input_embeddings().weight)
    similarity = prefix_embeddings @ voc_embeddings.T
    token_ids = similarity.argmax(dim=1)
    return tokenizer.decode(token_ids=token_ids)


@app.cell
def _():
    # _pretrained_model = "out/pt-multinerd-mmbert-base/checkpoint-355265"
    _pretrained_model = "jhu-clsp/mmBERT-base"
    tokenizer = AutoTokenizer.from_pretrained(_pretrained_model)
    model = AutoModel.from_pretrained(
        _pretrained_model,
        trust_remote_code=True,
        use_safetensors=True,
    )
    return model, tokenizer


@app.cell
def _(model, tokenizer):
    _prefix_tokens = " ".join(Multinerd._SYSTEM_TOKENS)
    _prefix_embeddings = encode_prefix(
        prefix=_prefix_tokens,
        model=model,
        tokenizer=tokenizer,
    )

    decode_prefix(
        prefix_embeddings=_prefix_embeddings,
        model=model,
        tokenizer=tokenizer,
    )
    return


if __name__ == "__main__":
    app.run()
