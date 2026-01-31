import logging
from typing import Annotated, Literal

from typer import Option, Typer

app = Typer(no_args_is_help=True, add_completion=False)
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


@app.command(help="Export tensorboard scalars to csv")
def export_tensorboard(logdir: str, outfile: str = "tb.csv"):
    from icft.scripts.export_tensorboard import main

    main(logdir=logdir, outfile=outfile)


@app.command(
    help="Fine tune a pretrained bert model for sequence classification on MultiNERD dataset"
)
def fine_tune_bert_multinerd(
    system_prompt: Literal["ner", "random", "none"] = "none",
    head_only: Annotated[
        bool, Option(help="If set freeze all parameters except classifier head")
    ] = False,
    pretrained_model: str = "distilbert-base-uncased",
    out_dir: str = "out/fine-tune",
    epochs: int = 1,
    train_split: str = "train",
    eval_split: str = "validation",
    test_split: str = "test",
):
    from icft.scripts.fine_tune_bert_multinerd import main

    main(
        system_prompt=system_prompt,
        head_only=head_only,
        pretrained_model=pretrained_model,
        out_dir=out_dir,
        epochs=epochs,
        train_split=train_split,
        eval_split=eval_split,
        test_split=test_split,
    )


@app.command(
    help="P-tune a pretrained bert model for sequence classification on MultiNERD dataset"
)
def prompt_tune_bert_multinerd(
    prefix_random_init: Annotated[
        bool,
        Option(
            help="If not set intialize prefix tensor from --pretrained_model embedding layer"
        ),
    ] = False,
    pretrained_model: str = "distilbert-base-uncased",
    out_dir: str = "out/prefix-tune",
    epochs: int = 1,
    train_split: str = "train",
    eval_split: str = "validation",
    test_split: str = "test",
):
    from icft.scripts.prompt_tune_bert_multinerd import main

    main(
        prefix_random_init=prefix_random_init,
        pretrained_model=pretrained_model,
        out_dir=out_dir,
        epochs=epochs,
        train_split=train_split,
        eval_split=eval_split,
        test_split=test_split,
    )


if __name__ == "__main__":
    app()
