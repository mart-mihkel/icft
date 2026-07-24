"""Typer CLI entry points for fine-tuning, prompt-tuning, few-shot eval, and metrics."""

from typing import Annotated

from typer import Option, Typer

from instruct.types import Architecture, DatasetName, LogLevel, PrefixInit

app = Typer(no_args_is_help=True)


ModelOption = Annotated[
    str,
    Option(help="HuggingFace model or path to checkpoint"),
]

DatasetOption = Annotated[
    DatasetName.__value__,
    Option(help="Dataset name"),
]

ArchOption = Annotated[
    Architecture.__value__ | None,
    Option(help="Override auto-detected model architecture"),
]

NShotOption = Annotated[
    int,
    Option(help="Number of examples in system prompt"),
]

NTrainSamplesOption = Annotated[
    int | None,
    Option(help="If present take a subset of tokenized train data"),
]

NDevSamplesOption = Annotated[
    int | None,
    Option(help="If present take a subset of tokenized dev data"),
]

DoEvalOption = Annotated[
    bool,
    Option(help="Run evalutaion during training"),
]

EarlyStoppingOption = Annotated[
    bool,
    Option(help="Stop training early if eval metrics don't improve"),
]

EpochsOption = Annotated[
    int,
    Option(help="Number of training epochs"),
]

BatchSizeOption = Annotated[
    int,
    Option(help="Training/eval batch size"),
]

LearningRateOption = Annotated[
    float,
    Option(help="Optimizer learning rate"),
]

ExperimentOption = Annotated[
    str,
    Option(help="Experiment for tracking"),
]

RunNameOption = Annotated[
    str | None,
    Option(help="Run name for tracking, inferred from parameters by default"),
]

LogLevelOption = Annotated[
    LogLevel.__value__,
    Option(help="Logging verbosity"),
]

SeedOption = Annotated[
    int | None,
    Option(help="Random seed"),
]


def _set_seed(seed: int) -> None:
    import random

    import numpy
    import torch

    random.seed(seed)
    numpy.random.seed(seed)  # noqa: NPY002 -- seeds the global state 3rd-party libs read
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@app.command(no_args_is_help=True, help="Fine-tune and run test evaluation")
def fine_tune(
    model: ModelOption,
    dataset: DatasetOption,
    *,
    arch: ArchOption = None,
    head_only: Annotated[
        bool,
        Option(help="Freeze all parameters except for classifier head"),
    ] = False,
    n_shot: NShotOption = 0,
    n_train_samples: NTrainSamplesOption = None,
    n_dev_samples: NDevSamplesOption = None,
    do_eval: DoEvalOption = False,
    early_stopping: EarlyStoppingOption = False,
    epochs: EpochsOption = 3,
    batch_size: BatchSizeOption = 8,
    learning_rate: LearningRateOption = 5e-5,
    experiment: ExperimentOption = "instruct",
    run_name: RunNameOption = None,
    log_level: LogLevelOption = "info",
    seed: SeedOption = None,
) -> None:
    """Fine-tune and run test evaluation."""
    from instruct.logging import logger
    from instruct.scripts.fine_tune import fine_tune

    if seed is not None:
        _set_seed(seed)

    logger.setLevel(log_level.upper())
    fine_tune(
        model_path=model,
        dataset=dataset,
        arch=arch,
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
    model: ModelOption,
    dataset: DatasetOption,
    prefix_init: Annotated[
        PrefixInit.__value__,
        Option(help="Prefix initialization method"),
    ],
    *,
    arch: ArchOption = None,
    n_shot: NShotOption = 0,
    n_train_samples: NTrainSamplesOption = None,
    n_dev_samples: NDevSamplesOption = None,
    do_eval: DoEvalOption = False,
    early_stopping: EarlyStoppingOption = False,
    epochs: EpochsOption = 3,
    batch_size: BatchSizeOption = 8,
    learning_rate: LearningRateOption = 1e-3,
    experiment: ExperimentOption = "instruct",
    run_name: RunNameOption = None,
    log_level: LogLevelOption = "info",
    seed: SeedOption = None,
) -> None:
    """Prompt-tune and run test evaluation."""
    from instruct.logging import logger
    from instruct.scripts.prompt_tune import prompt_tune

    if seed is not None:
        _set_seed(seed)

    logger.setLevel(log_level.upper())
    prompt_tune(
        model_path=model,
        dataset=dataset,
        prefix_init=prefix_init,
        arch=arch,
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
    model: ModelOption,
    dataset: DatasetOption,
    *,
    arch: ArchOption = None,
    n_shot: NShotOption = 5,
    batch_size: BatchSizeOption = 8,
    experiment: ExperimentOption = "instruct",
    run_name: RunNameOption = None,
    log_level: LogLevelOption = "info",
    seed: SeedOption = None,
) -> None:
    """Run test evaluation with few-shot learning."""
    from instruct.logging import logger
    from instruct.scripts.few_shot import few_shot

    if seed is not None:
        _set_seed(seed)

    logger.setLevel(log_level.upper())
    few_shot(
        model_path=model,
        arch=arch,
        dataset=dataset,
        n_shot=n_shot,
        batch_size=batch_size,
        experiment=experiment,
        run_name=run_name,
    )


@app.command(no_args_is_help=True, help="Export MLflow experiments to csv")
def collect_metrics(
    experiment: ExperimentOption = "instruct",
    mlflow_tracking_uri: Annotated[
        str,
        Option(
            help="Can be overriden with envrionment variables",
            envvar="MLFLOW_TRACKING_URI",
        ),
    ] = "sqlite:///mlflow.db",
    log_level: LogLevelOption = "info",
) -> None:
    """Export MLflow experiments to csv."""
    from instruct.logging import logger
    from instruct.scripts.tracking import collect_metrics

    logger.setLevel(log_level.upper())
    collect_metrics(experiment, mlflow_tracking_uri, write_csv=True)


if __name__ == "__main__":
    app()
