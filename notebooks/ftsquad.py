import marimo

__generated_with = "0.18.1"
app = marimo.App(width="medium")

with app.setup:
    import logging
    from pathlib import Path

    import torch
    from transformers import (
        ModernBertForQuestionAnswering,
        PreTrainedTokenizerFast,
        default_data_collator,
    )

    from cptlms.squad import Squad
    from cptlms.trainer import Trainer

    logging.basicConfig(level="INFO")
    torch.set_float32_matmul_precision("high")


@app.cell
def _():
    _pretrained_model = "jhu-clsp/mmBERT-small"
    _tokenizer = PreTrainedTokenizerFast.from_pretrained(_pretrained_model)
    squad = Squad(_tokenizer)
    model = ModernBertForQuestionAnswering.from_pretrained(_pretrained_model)
    return model, squad


@app.cell
def _(model, squad):
    def _drop_offset_mappings(batch):
        for item in batch:
            item.pop("offset_mapping", None)

        return default_data_collator(batch)

    trainer = Trainer(
        model=model,
        epochs=10,
        qa_dataset=squad,
        collate_fn=_drop_offset_mappings,
        batch_size=32,
        out_dir=Path("out/mmbert-small-ft"),
    )
    return (trainer,)


@app.cell
def _(trainer):
    trainer.train()
    return


@app.cell
def _(trainer):
    trainer.telemetry
    return


if __name__ == "__main__":
    app.run()
