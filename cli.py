import logging

import typer

logging.basicConfig(level="INFO")
logger = logging.getLogger(__name__)

app = typer.Typer()


@app.command()
def fine_tune(
    pretrained_model: str = "jhu-clsp/mmBERT-small",
    out_dir: str = "out/mmbert-small-ft",
    epochs: int = 10,
    batch_size: int = 32,
):
    from pathlib import Path

    import torch
    from transformers import (
        ModernBertForQuestionAnswering,
        PreTrainedTokenizerFast,
        default_data_collator,
    )

    from cptlms.squad import SQuAD
    from cptlms.trainer import Trainer

    torch.set_float32_matmul_precision("high")

    out_path = Path(out_dir)

    tokenizer = PreTrainedTokenizerFast.from_pretrained(pretrained_model)
    model = ModernBertForQuestionAnswering.from_pretrained(pretrained_model)

    squad = SQuAD(tokenizer)

    def _drop_offset_mappings_collator(batch):
        for item in batch:
            item.pop("offset_mapping", None)

        return default_data_collator(batch)

    trainer = Trainer(
        model=model,
        epochs=epochs,
        qa_dataset=squad,
        collate_fn=_drop_offset_mappings_collator,
        batch_size=batch_size,
        out_dir=out_path,
    )

    trainer.train()


@app.command()
def p_tune():
    raise NotImplementedError()


if __name__ == "__main__":
    app()
