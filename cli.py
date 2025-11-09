import typer

app = typer.Typer()


@app.command()
def bert(
    pretrained_model: str = "bert-base-cased",
    num_virtual_tokens: int = 32,
    trainer_output_dir: str = "out/bert",
    num_epochs: int = 3,
    batch_size: int = 16,
):
    from tuning.bert import main

    main(
        pretrained_model,
        num_virtual_tokens,
        trainer_output_dir,
        num_epochs,
        batch_size,
    )


if __name__ == "__main__":
    app()
