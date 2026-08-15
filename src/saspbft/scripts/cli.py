"""CLI entry points."""

from collections.abc import Callable
from typing import get_args

from click import Choice, FloatRange, IntRange, group, option

from saspbft.logging import setup_logging
from saspbft.types import Architecture, DatasetName, LogLevel, PrefixInit

type ClickDecorator = Callable[[Callable[..., None]], Callable[..., None]]


@group(
    help="CLI interface for running scripts related to the study",
    context_settings={"help_option_names": ["-h", "--help"]},
)
def app() -> None:
    """CLI interface for running scripts related to the study."""
    setup_logging()


model_option = option(
    "--model",
    "-m",
    required=True,
    help="HuggingFace model or path to checkpoint",
)

dataset_option = option(
    "--dataset",
    "-d",
    type=Choice(get_args(DatasetName.__value__)),
    required=True,
    help="Dataset name",
)

arch_option = option(
    "--arch",
    "-a",
    type=Choice(get_args(Architecture.__value__)),
    default=None,
    help="Override auto-detected model architecture",
)

head_only_option = option(
    "--head-only",
    is_flag=True,
    default=False,
    help="Freeze all parameters except for classifier head",
)

prefix_init_option = option(
    "--prefix-init",
    "-p",
    type=Choice(get_args(PrefixInit.__value__)),
    required=True,
    help="Prefix initialization method",
)

n_shot_option = option(
    "--n-shot",
    "-n",
    type=IntRange(min=0),
    default=0,
    show_default=True,
    help="Number of examples in system prompt",
)

epochs_option = option(
    "--epochs",
    "-e",
    type=IntRange(min=0),
    default=5,
    show_default=True,
    help="Number of training epochs",
)

train_samples_option = option(
    "--train-samples",
    "-t",
    type=IntRange(min=0),
    default=None,
    help="If present take a subset of tokenized train data",
)

val_samples_option = option(
    "--val-samples",
    "-v",
    type=IntRange(min=0),
    default=None,
    help="If present take a subset of tokenized validation data",
)

do_eval_option = option(
    "--do-eval",
    is_flag=True,
    default=False,
    help="Run evalutaion during training",
)

early_stopping_option = option(
    "--early-stopping",
    is_flag=True,
    default=False,
    help="Stop training early if eval metrics don't improve",
)

resume_option = option(
    "--resume/--no-resume",
    default=True,
    show_default=True,
    help="Continue from the last checkpoint of a run with the same name",
)

batch_size_option = option(
    "--batch-size",
    "-b",
    type=IntRange(min=1),
    default=8,
    show_default=True,
    help="Training/eval batch size",
)

tracking_uri_option = option(
    "--mlflow-tracking-uri",
    envvar="MLFLOW_TRACKING_URI",
    default="sqlite:///mlflow.db",
    show_default=True,
    help="Can be overriden with envrionment variables",
)

experiment_option = option(
    "--mlflow-experiment",
    default="saspbft",
    show_default=True,
    help="Experiment for tracking",
)

run_name_option = option(
    "--mlflow-run-name",
    default=None,
    help="Run name for tracking, inferred from parameters by default",
)

metric_option = option(
    "--metric",
    "metrics",
    multiple=True,
    default=(),
    show_default=True,
    help="Per-step metric to export the history of, repeatable",
)

log_level_option = option(
    "--log-level",
    "-l",
    type=Choice(get_args(LogLevel.__value__)),
    default="info",
    show_default=True,
    help="Log level",
)

seed_option = option(
    "--seed",
    type=IntRange(min=0),
    default=None,
    help="Random seed",
)

job_option = option(
    "--job",
    "-j",
    help="Predefined job name or regex to submit to SLURM",
)

list_jobs_option = option(
    "--list-jobs",
    is_flag=True,
    default=False,
    help="List all predefined jobs and exit",
)


def learning_rate_option(default: float) -> ClickDecorator:
    """Build a `--learning-rate` option with a command-specific default."""
    return option(
        "--learning-rate",
        "-r",
        type=FloatRange(min=0),
        default=default,
        show_default=True,
        help="Optimizer learning rate",
    )


def _assert_torch_installed() -> None:
    import sys
    from importlib.util import find_spec

    from saspbft.logging import logger

    spec = find_spec("torch")
    if spec is not None:
        logger.debug("pytorch is installed")
        return

    logger.error("pytorch is not installed")
    logger.error("install pytorch using `uv sync --extra [cpu|cu132]`")
    sys.exit(1)


def _set_seed(seed: int | None) -> None:
    import random

    import numpy as np
    import torch

    from saspbft.logging import logger

    if seed is None:
        logger.warning("randomness is not fixed")
        return

    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@app.command(help="Fine-tune and run test evaluation")
@model_option
@dataset_option
@arch_option
@head_only_option
@n_shot_option
@train_samples_option
@val_samples_option
@do_eval_option
@early_stopping_option
@resume_option
@epochs_option
@batch_size_option
@learning_rate_option(default=5e-5)
@experiment_option
@run_name_option
@tracking_uri_option
@log_level_option
@seed_option
def fine_tune(
    model: str,
    dataset: DatasetName.__value__,
    *,
    arch: Architecture.__value__ | None,
    head_only: bool,
    n_shot: int,
    train_samples: int | None,
    val_samples: int | None,
    do_eval: bool,
    early_stopping: bool,
    resume: bool,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    mlflow_experiment: str,
    mlflow_run_name: str | None,
    mlflow_tracking_uri: str,
    log_level: LogLevel.__value__,
    seed: int | None,
) -> None:
    """Fine-tune and run test evaluation."""
    from saspbft.logging import logger
    from saspbft.scripts.fine import fine_tune

    logger.setLevel(log_level.upper())
    _assert_torch_installed()
    _set_seed(seed)

    logger.debug("finished preamble")
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
        resume=resume,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        mlflow_experiment=mlflow_experiment,
        mlflow_run_name=mlflow_run_name,
        mlflow_tracking_uri=mlflow_tracking_uri,
    )


@app.command(help="Prompt-tune and run test evaluation")
@model_option
@dataset_option
@prefix_init_option
@arch_option
@n_shot_option
@train_samples_option
@val_samples_option
@do_eval_option
@early_stopping_option
@resume_option
@epochs_option
@batch_size_option
@learning_rate_option(default=1e-3)
@experiment_option
@run_name_option
@tracking_uri_option
@log_level_option
@seed_option
def prompt_tune(
    model: str,
    dataset: DatasetName.__value__,
    prefix_init: PrefixInit.__value__,
    *,
    arch: Architecture.__value__ | None,
    n_shot: int,
    train_samples: int | None,
    val_samples: int | None,
    do_eval: bool,
    early_stopping: bool,
    resume: bool,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    mlflow_experiment: str,
    mlflow_run_name: str | None,
    mlflow_tracking_uri: str,
    log_level: LogLevel.__value__,
    seed: int | None,
) -> None:
    """Prompt-tune and run test evaluation."""
    from saspbft.logging import logger
    from saspbft.scripts.prompt import prompt_tune

    logger.setLevel(log_level.upper())
    _assert_torch_installed()
    _set_seed(seed)

    logger.debug("finished preamble")
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
        resume=resume,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        mlflow_experiment=mlflow_experiment,
        mlflow_run_name=mlflow_run_name,
        mlflow_tracking_uri=mlflow_tracking_uri,
    )


@app.command(help="Run test evaluation with few-shot learning")
@model_option
@dataset_option
@arch_option
@n_shot_option
@batch_size_option
@experiment_option
@run_name_option
@tracking_uri_option
@log_level_option
@seed_option
def few_shot(
    model: str,
    dataset: DatasetName.__value__,
    *,
    arch: Architecture.__value__ | None,
    n_shot: int,
    batch_size: int,
    mlflow_experiment: str,
    mlflow_run_name: str | None,
    mlflow_tracking_uri: str,
    log_level: LogLevel.__value__,
    seed: int | None,
) -> None:
    """Run test evaluation with few-shot learning."""
    from saspbft.logging import logger
    from saspbft.scripts.fewshot import few_shot

    logger.setLevel(log_level.upper())
    _assert_torch_installed()
    _set_seed(seed)

    logger.debug("finished preamble")
    few_shot(
        model_path=model,
        arch=arch,
        dataset=dataset,
        n_shot=n_shot,
        batch_size=batch_size,
        mlflow_experiment=mlflow_experiment,
        mlflow_run_name=mlflow_run_name,
        mlflow_tracking_uri=mlflow_tracking_uri,
    )


@app.command(help="Submit SLURM jobs for each model in a predefined configuration")
@job_option
@list_jobs_option
@log_level_option
def submit(job: str | None, *, list_jobs: bool, log_level: LogLevel.__value__) -> None:
    """Submit SLURM jobs for each model in a predefined configuration."""
    import sys

    from saspbft.logging import logger
    from saspbft.scripts.submit import show, submit

    logger.setLevel(log_level.upper())
    if list_jobs:
        show()
    elif job is not None:
        _assert_torch_installed()
        submit(job)
    else:
        logger.error("one of '--job' or '--list-jobs' must be present")
        sys.exit(1)


@app.command(help="Export MLflow experiments to csv")
@experiment_option
@tracking_uri_option
@metric_option
@log_level_option
def collect(
    mlflow_experiment: str,
    mlflow_tracking_uri: str,
    metrics: tuple[str, ...],
    log_level: LogLevel.__value__,
) -> None:
    """Export MLflow experiments to csv."""
    from saspbft.logging import logger
    from saspbft.scripts.tracking import collect_runs

    logger.setLevel(log_level.upper())
    collect_runs(mlflow_experiment, mlflow_tracking_uri, metrics, write_csv=True)


if __name__ == "__main__":
    app()
