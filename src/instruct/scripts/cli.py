from __future__ import annotations

from typing import Annotated, Literal

from typer import Option, Typer

from instruct.types import DatasetName, PrefixInit

app = Typer(no_args_is_help=True)


def _set_seed(seed: int) -> None:
    import random

    import numpy
    import torch

    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@app.command(no_args_is_help=True, help="Fine-tune and run test evaluation")
def fine_tune(
    model: Annotated[str, Option(help="HuggingFace model or path to checkpoint")],
    dataset: Annotated[DatasetName.__value__, Option(help="Dataset name")],
    head_only: Annotated[
        bool,
        Option(help="Freeze all parameters except for classifier head"),
    ] = False,
    n_shot: Annotated[int, Option(help="Number of examples in system prompt")] = 0,
    n_train_samples: Annotated[
        int | None,
        Option(help="If present take a subset of tokenized train data"),
    ] = None,
    n_dev_samples: Annotated[
        int | None,
        Option(help="If present take a subset of tokenized dev data"),
    ] = None,
    do_eval: Annotated[bool, Option(help="Run evalutaion during training")] = False,
    early_stopping: Annotated[
        bool,
        Option(help="Stop training early if eval metrics don't improve"),
    ] = False,
    epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 5e-5,
    experiment: Annotated[str, Option(help="Experiment for tracking")] = "instruct",
    run_name: Annotated[
        str | None,
        Option(help="Run name for tracking, inferred from parameters by default"),
    ] = None,
    log_level: Literal["debug", "info", "warning", "error"] = "info",
    seed: Annotated[int | None, Option(help="Random seed")] = None,
) -> None:
    from instruct.logging import logger
    from instruct.scripts.fine_tune import fine_tune

    if seed is not None:
        _set_seed(seed)

    logger.setLevel(log_level.upper())
    fine_tune(
        model_path=model,
        dataset=dataset,
        head_only=head_only,
        n_shot=n_shot,
        n_train_samples=n_train_samples,
        n_dev_samples=n_dev_samples,
        do_eval=do_eval,
        early_stopping=early_stopping,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        experiment=experiment,
        run_name=run_name,
    )


@app.command(no_args_is_help=True, help="Prompt-tune and run test evaluation")
def prompt_tune(
    model: Annotated[str, Option(help="HuggingFace model or path to checkpoint")],
    dataset: Annotated[DatasetName.__value__, Option(help="Dataset name")],
    prefix_init: Annotated[
        PrefixInit.__value__,
        Option(help="Prefix initialization method"),
    ],
    n_shot: Annotated[int, Option(help="Number of examples in system prompt")] = 0,
    n_train_samples: Annotated[
        int | None,
        Option(help="If present take a subset of tokenized train data"),
    ] = None,
    n_dev_samples: Annotated[
        int | None,
        Option(help="If present take a subset of tokenized dev data"),
    ] = None,
    do_eval: Annotated[bool, Option(help="Run evalutaion during training")] = False,
    early_stopping: Annotated[
        bool,
        Option(help="Stop training early if eval metrics don't improve"),
    ] = False,
    epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 1e-3,
    experiment: Annotated[str, Option(help="Experiment for tracking")] = "instruct",
    run_name: Annotated[
        str | None,
        Option(help="Run name for tracking, inferred from parameters by default"),
    ] = None,
    log_level: Literal["debug", "info", "warning", "error"] = "info",
    seed: Annotated[int | None, Option(help="Random seed")] = None,
) -> None:
    from instruct.logging import logger
    from instruct.scripts.prompt_tune import prompt_tune

    if seed is not None:
        _set_seed(seed)

    logger.setLevel(log_level.upper())
    prompt_tune(
        model_path=model,
        dataset=dataset,
        prefix_init=prefix_init,
        n_shot=n_shot,
        n_train_samples=n_train_samples,
        n_dev_samples=n_dev_samples,
        do_eval=do_eval,
        early_stopping=early_stopping,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        experiment=experiment,
        run_name=run_name,
    )


@app.command(no_args_is_help=True, help="Run test evaluation with few-shot learning")
def few_shot(
    model: Annotated[str, Option(help="HuggingFace model or path to checkpoint")],
    dataset: Annotated[DatasetName.__value__, Option(help="Dataset name")],
    n_shot: Annotated[int, Option(help="Number of examples in system prompt")] = 5,
    batch_size: int = 8,
    experiment: Annotated[str, Option(help="Experiment for tracking")] = "instruct",
    run_name: Annotated[
        str | None,
        Option(help="Run name for tracking, inferred from parameters by default"),
    ] = None,
    log_level: Literal["debug", "info", "warning", "error"] = "info",
    seed: Annotated[int | None, Option(help="Random seed")] = None,
) -> None:
    from instruct.logging import logger
    from instruct.scripts.few_shot import few_shot

    if seed is not None:
        _set_seed(seed)

    logger.setLevel(log_level.upper())
    few_shot(
        model_path=model,
        dataset=dataset,
        n_shot=n_shot,
        batch_size=batch_size,
        experiment=experiment,
        run_name=run_name,
    )


@app.command(no_args_is_help=True, help="Export MLflow experiments to csv")
def collect_metrics(
    experiment: Annotated[str, Option(help="MLflow experiment name")] = "instruct",
    mlflow_tracking_uri: Annotated[
        str,
        Option(
            help="Can be overriden with envrionment variables",
            envvar="MLFLOW_TRACKING_URI",
        ),
    ] = "sqlite:///mlflow.db",
    log_level: Literal["debug", "info", "warning", "error"] = "info",
) -> None:
    from instruct.logging import logger
    from instruct.scripts.tracking import collect_metrics

    logger.setLevel(log_level.upper())
    collect_metrics(experiment, mlflow_tracking_uri, write_csv=True)


if __name__ == "__main__":
    app()
