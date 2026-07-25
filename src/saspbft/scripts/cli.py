"""Typer CLI entry points for fine-tuning, prompt-tuning, few-shot eval, and metrics."""

from typing import Annotated

from typer import Option, Typer

from saspbft.types import Architecture, DatasetName, LogLevel, PrefixInit

app = Typer(no_args_is_help=True)


ModelOption = Annotated[
    str,
    Option("--model", "-m", help="HuggingFace model or path to checkpoint"),
]

DatasetOption = Annotated[
    DatasetName.__value__,
    Option("--dataset", "-d", help="Dataset name"),
]

ArchOption = Annotated[
    Architecture.__value__ | None,
    Option("--arch", "-a", help="Override auto-detected model architecture"),
]

HeadOnlyOption = Annotated[
    bool,
    Option(
        "--head-only",
        help="Freeze all parameters except for classifier head",
    ),
]

PrefixInitOption = Annotated[
    PrefixInit.__value__,
    Option("--prefix-init", "-p", help="Prefix initialization method"),
]

NShotOption = Annotated[
    int,
    Option(
        "--n-shot",
        "-n",
        min=0,
        help="Number of examples in system prompt",
    ),
]

TrainSamplesOption = Annotated[
    int | None,
    Option(
        "--train-samples",
        "-t",
        min=0,
        help="If present take a subset of tokenized train data",
    ),
]

ValSamplesOption = Annotated[
    int | None,
    Option(
        "--val-samples",
        "-v",
        min=0,
        help="If present take a subset of tokenized validation data",
    ),
]

DoEvalOption = Annotated[
    bool,
    Option("--do-eval", help="Run evalutaion during training"),
]

EarlyStoppingOption = Annotated[
    bool,
    Option(
        "--early-stopping",
        help="Stop training early if eval metrics don't improve",
    ),
]

EpochsOption = Annotated[
    int,
    Option("--epochs", "-e", min=0, help="Number of training epochs"),
]

BatchSizeOption = Annotated[
    int,
    Option("--batch-size", "-b", min=1, help="Training/eval batch size"),
]

LearningRateOption = Annotated[
    float,
    Option("--learning-rate", "-l", min=0, help="Optimizer learning rate"),
]

TrackingURIOption = Annotated[
    str,
    Option(
        "--mlflow-tracking-uri",
        help="Can be overriden with envrionment variables",
        envvar="MLFLOW_TRACKING_URI",
    ),
]

ExperimentOption = Annotated[
    str,
    Option("--experiment", "-x", help="Experiment for tracking"),
]

RunNameOption = Annotated[
    str | None,
    Option(
        "--run-name",
        "-r",
        help="Run name for tracking, inferred from parameters by default",
    ),
]

LogLevelOption = Annotated[
    LogLevel.__value__,
    Option("--log-level", help="Logging verbosity"),
]

SeedOption = Annotated[
    int | None,
    Option("--seed", min=0, help="Random seed"),
]

JobOption = Annotated[
    str,
    Option("--job", "-j", help="Predefined job to submit to SLURM"),
]


def _set_seed(seed: int) -> None:
    import random

    import numpy
    import torch

    random.seed(seed)
    numpy.random.seed(seed)  # noqa: NPY002
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@app.command(no_args_is_help=True, help="Fine-tune and run test evaluation")
def fine_tune(
    model: ModelOption,
    dataset: DatasetOption,
    *,
    arch: ArchOption = None,
    head_only: HeadOnlyOption = False,
    n_shot: NShotOption = 0,
    train_samples: TrainSamplesOption = None,
    val_samples: ValSamplesOption = None,
    do_eval: DoEvalOption = False,
    early_stopping: EarlyStoppingOption = False,
    epochs: EpochsOption = 3,
    batch_size: BatchSizeOption = 8,
    learning_rate: LearningRateOption = 5e-5,
    experiment: ExperimentOption = "saspbft",
    run_name: RunNameOption = None,
    log_level: LogLevelOption = "info",
    seed: SeedOption = None,
) -> None:
    """Fine-tune and run test evaluation."""
    from saspbft.logging import logger
    from saspbft.scripts.fine import fine_tune

    if seed is not None:
        _set_seed(seed)

    logger.setLevel(log_level.upper())
    fine_tune(
        model_path=model,
        dataset=dataset,
        arch=arch,
        head_only=head_only,
        n_shot=n_shot,
        train_samples=train_samples,
        val_samples=val_samples,
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
    prefix_init: PrefixInitOption,
    *,
    arch: ArchOption = None,
    n_shot: NShotOption = 0,
    train_samples: TrainSamplesOption = None,
    val_samples: ValSamplesOption = None,
    do_eval: DoEvalOption = False,
    early_stopping: EarlyStoppingOption = False,
    epochs: EpochsOption = 3,
    batch_size: BatchSizeOption = 8,
    learning_rate: LearningRateOption = 1e-3,
    experiment: ExperimentOption = "saspbft",
    run_name: RunNameOption = None,
    log_level: LogLevelOption = "info",
    seed: SeedOption = None,
) -> None:
    """Prompt-tune and run test evaluation."""
    from saspbft.logging import logger
    from saspbft.scripts.prompt import prompt_tune

    if seed is not None:
        _set_seed(seed)

    logger.setLevel(log_level.upper())
    prompt_tune(
        model_path=model,
        dataset=dataset,
        prefix_init=prefix_init,
        arch=arch,
        n_shot=n_shot,
        train_samples=train_samples,
        val_samples=val_samples,
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
    experiment: ExperimentOption = "saspbft",
    run_name: RunNameOption = None,
    log_level: LogLevelOption = "info",
    seed: SeedOption = None,
) -> None:
    """Run test evaluation with few-shot learning."""
    from saspbft.logging import logger
    from saspbft.scripts.fewshot import few_shot

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


@app.command(
    no_args_is_help=True,
    help="Submit SLURM jobs for each model in a predefined configuration",
)
def submit(
    job: JobOption,
    log_level: LogLevelOption = "info",
) -> None:
    """Submit SLURM jobs for each model in a predefined configuration."""
    from saspbft.logging import logger
    from saspbft.scripts.submit import submit

    logger.setLevel(log_level.upper())
    submit(job)


@app.command(no_args_is_help=True, help="Export MLflow experiments to csv")
def collect(
    experiment: ExperimentOption = "saspbft",
    mlflow_tracking_uri: TrackingURIOption = "sqlite:///mlflow.db",
    log_level: LogLevelOption = "info",
) -> None:
    """Export MLflow experiments to csv."""
    from saspbft.logging import logger
    from saspbft.scripts.tracking import collect_metrics

    logger.setLevel(log_level.upper())
    collect_metrics(experiment, mlflow_tracking_uri, write_csv=True)


if __name__ == "__main__":
    app()
