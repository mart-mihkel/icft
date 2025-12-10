import logging

import typer

app = typer.Typer(add_completion=False)
logger = logging.getLogger("cptlms")


def _setup_logging(out_dir: str):
    import os
    import sys
    from datetime import datetime
    from logging import FileHandler, StreamHandler

    os.makedirs(out_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%d%m%Y-%H%M%S")
    log_path = f"{out_dir}/logs-{timestamp}.log"
    handlers = [StreamHandler(sys.stdout), FileHandler(log_path)]
    logging.basicConfig(level=logging.INFO, handlers=handlers)
    logger.info("set logger file handler to %s", log_path)


@app.command()
def fine_tune(
    pretrained_model: str = "bert-base-uncased",
    out_dir: str = "out/ft",
    epochs: int = 5,
    batch_size: int = 32,
):
    from pathlib import Path

    import torch
    from transformers import AutoModelForQuestionAnswering, AutoTokenizer

    from cptlms.squad import Squad
    from cptlms.trainer import Trainer

    _setup_logging(out_dir=out_dir)
    torch.set_float32_matmul_precision("high")

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
    squad = Squad(tokenizer)

    model = AutoModelForQuestionAnswering.from_pretrained(pretrained_model)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("total parameters:     %d", total_params)
    logger.info("trainable parameters: %d", trainable_params)

    trainer = Trainer(
        model=model,
        epochs=epochs,
        qa_dataset=squad,
        batch_size=batch_size,
        collate_fn=Squad.collate_fn,
        out_dir=Path(out_dir),
    )

    trainer.train()


@app.command()
def p_tune(
    pretrained_model: str = "bert-base-uncased",
    out_dir: str = "out/pt",
    epochs: int = 20,
    batch_size: int = 32,
    num_virtual_tokens: int = 32,
):
    from pathlib import Path

    import torch
    from transformers import AutoModelForQuestionAnswering, AutoTokenizer

    from cptlms.bert import PTuningBert
    from cptlms.squad import Squad
    from cptlms.trainer import Trainer

    _setup_logging(out_dir=out_dir)
    torch.set_float32_matmul_precision("high")

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
    squad = Squad(tokenizer)

    base_bert = AutoModelForQuestionAnswering.from_pretrained(pretrained_model)
    pt_bert = PTuningBert(bert=base_bert, num_virtual_tokens=num_virtual_tokens)

    total_params = sum(p.numel() for p in pt_bert.parameters())
    trainable_params = sum(p.numel() for p in pt_bert.parameters() if p.requires_grad)
    logger.info("total parameters:     %d", total_params)
    logger.info("trainable parameters: %d", trainable_params)

    trainer = Trainer(
        model=pt_bert,
        epochs=epochs,
        qa_dataset=squad,
        batch_size=batch_size,
        collate_fn=Squad.collate_fn,
        out_dir=Path(out_dir),
    )

    trainer.train()


if __name__ == "__main__":
    app()
