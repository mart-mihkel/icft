"""Predefined SLURM jobs."""

import shlex
from typing import TYPE_CHECKING, Literal, NamedTuple

from saspbft.constants import SLURMDIR

if TYPE_CHECKING:
    from saspbft.types import Architecture, DatasetName, LogLevel, PrefixInit


class _FineTuneArgs(NamedTuple):
    """Parameters for a single `uv run --no-sync cli fine-tune ...` invocation."""

    dataset: DatasetName
    head_only: bool
    arch: Architecture | None = None
    epochs: int = 5
    batch_size: int = 8
    learning_rate: float = 5e-5
    n_shot: int = 0
    do_eval: bool = False
    early_stopping: bool = False
    resume: bool = True
    train_samples: int | None = None
    val_samples: int | None = None
    mlflow_run_name: str | None = None
    mlflow_experiment: str = "saspbft"
    mlflow_tracking_uri: str | None = None
    log_level: LogLevel = "debug"
    seed: int = 42
    command: Literal["fine-tune"] = "fine-tune"


class _PromptTuneArgs(NamedTuple):
    """Parameters for a single `uv run --no-sync cli prompt-tune ...` invocation."""

    dataset: DatasetName
    prefix_init: PrefixInit
    arch: Architecture | None = None
    epochs: int = 5
    batch_size: int = 8
    learning_rate: float = 1e-3
    n_shot: int = 0
    do_eval: bool = False
    early_stopping: bool = False
    resume: bool = True
    train_samples: int | None = None
    val_samples: int | None = None
    mlflow_run_name: str | None = None
    mlflow_experiment: str = "saspbft"
    mlflow_tracking_uri: str | None = None
    log_level: LogLevel = "debug"
    seed: int = 42
    command: Literal["prompt-tune"] = "prompt-tune"


class _FewShotArgs(NamedTuple):
    """Parameters for a single `uv run --no-sync cli few-shot ...` invocation."""

    dataset: DatasetName
    arch: Architecture | None = None
    n_shot: int = 0
    batch_size: int = 8
    mlflow_run_name: str | None = None
    mlflow_experiment: str = "saspbft"
    mlflow_tracking_uri: str | None = None
    log_level: LogLevel = "debug"
    seed: int = 42
    command: Literal["few-shot"] = "few-shot"


type _Cli = _FineTuneArgs | _PromptTuneArgs | _FewShotArgs


class _Job(NamedTuple):
    """SLURM job paramerts."""

    job_name: str
    time: str
    mem: str
    cpus: int
    gres: str
    models: tuple[str, ...]
    cli: _Cli
    partition: str = "gpu"
    requeue: bool = True
    signal: str | None = None
    """Signal sent before the time limit runs out, e.g. `USR1@300`."""


JOBS: list[_Job] = [
    _Job(
        job_name="distilbert-head-only",
        time="00:10:00",
        mem="16GB",
        cpus=32,
        gres="gpu:h200-141g:1",
        models=("distilbert/distilbert-base-cased",),
        cli=_FineTuneArgs(
            dataset="multinerd",
            arch="encoder",
            head_only=True,
            train_samples=20000,
            val_samples=1024,
        ),
    ),
    _Job(
        job_name="distilbert-fine-tune",
        time="00:15:00",
        mem="16GB",
        cpus=32,
        gres="gpu:h200-141g:1",
        models=("distilbert/distilbert-base-cased",),
        cli=_FineTuneArgs(
            dataset="multinerd",
            arch="encoder",
            head_only=False,
            train_samples=20000,
            val_samples=1024,
        ),
    ),
    _Job(
        job_name="distilbert-prompt-tune-pretrained",
        time="00:15:00",
        mem="16GB",
        cpus=32,
        gres="gpu:h200-141g:1",
        models=("distilbert/distilbert-base-cased",),
        cli=_PromptTuneArgs(
            dataset="multinerd",
            prefix_init="pretrained",
            arch="encoder",
            train_samples=20000,
            val_samples=1024,
        ),
    ),
    _Job(
        job_name="distilbert-prompt-tune-random",
        time="00:15:00",
        mem="16GB",
        cpus=32,
        gres="gpu:h200-141g:1",
        models=("distilbert/distilbert-base-cased",),
        cli=_PromptTuneArgs(
            dataset="multinerd",
            prefix_init="random",
            arch="encoder",
            train_samples=20000,
            val_samples=1024,
        ),
    ),
    _Job(
        job_name="mmbert-head-only",
        time="00:30:00",
        mem="16GB",
        cpus=32,
        gres="gpu:h200-141g:1",
        models=(
            "jhu-clsp/mmBERT-small",
            "jhu-clsp/mmBERT-base",
        ),
        cli=_FineTuneArgs(
            dataset="multinerd",
            arch="encoder",
            head_only=True,
            train_samples=20000,
            val_samples=1024,
        ),
    ),
    _Job(
        job_name="mmbert-fine-tune",
        time="00:30:00",
        mem="16GB",
        cpus=32,
        gres="gpu:h200-141g:1",
        models=(
            "jhu-clsp/mmBERT-small",
            "jhu-clsp/mmBERT-base",
        ),
        cli=_FineTuneArgs(
            dataset="multinerd",
            arch="encoder",
            head_only=False,
            train_samples=20000,
            val_samples=1024,
        ),
    ),
    _Job(
        job_name="mmbert-prompt-tune-pretrained",
        time="00:30:00",
        mem="16GB",
        cpus=32,
        gres="gpu:h200-141g:1",
        models=(
            "jhu-clsp/mmBERT-small",
            "jhu-clsp/mmBERT-base",
        ),
        cli=_PromptTuneArgs(
            dataset="multinerd",
            prefix_init="pretrained",
            arch="encoder",
            train_samples=20000,
            val_samples=1024,
        ),
    ),
    _Job(
        job_name="mmbert-prompt-tune-random",
        time="00:30:00",
        mem="16GB",
        cpus=32,
        gres="gpu:h200-141g:1",
        models=(
            "jhu-clsp/mmBERT-small",
            "jhu-clsp/mmBERT-base",
        ),
        cli=_PromptTuneArgs(
            dataset="multinerd",
            prefix_init="random",
            arch="encoder",
            train_samples=20000,
            val_samples=1024,
        ),
    ),
    _Job(
        job_name="eurobert-head-only",
        time="01:00:00",
        mem="16GB",
        cpus=32,
        gres="gpu:h200-141g:1",
        models=(
            "EuroBERT/EuroBERT-210m",
            "EuroBERT/EuroBERT-610m",
            "EuroBERT/EuroBERT-2.1B",
        ),
        cli=_FineTuneArgs(
            dataset="multinerd",
            arch="encoder",
            head_only=True,
            train_samples=20000,
            val_samples=1024,
        ),
    ),
    _Job(
        job_name="eurobert-fine-tune",
        time="01:00:00",
        mem="16GB",
        cpus=32,
        gres="gpu:h200-141g:1",
        models=(
            "EuroBERT/EuroBERT-210m",
            "EuroBERT/EuroBERT-610m",
            "EuroBERT/EuroBERT-2.1B",
        ),
        cli=_FineTuneArgs(
            dataset="multinerd",
            arch="encoder",
            head_only=False,
            train_samples=20000,
            val_samples=1024,
        ),
    ),
    _Job(
        job_name="eurobert-prompt-tune-pretrained",
        time="01:00:00",
        mem="16GB",
        cpus=32,
        gres="gpu:h200-141g:1",
        models=(
            "EuroBERT/EuroBERT-210m",
            "EuroBERT/EuroBERT-610m",
            "EuroBERT/EuroBERT-2.1B",
        ),
        cli=_PromptTuneArgs(
            dataset="multinerd",
            prefix_init="pretrained",
            arch="encoder",
            train_samples=20000,
            val_samples=1024,
        ),
    ),
    _Job(
        job_name="eurobert-prompt-tune-random",
        time="01:00:00",
        mem="16GB",
        cpus=32,
        gres="gpu:h200-141g:1",
        models=(
            "EuroBERT/EuroBERT-210m",
            "EuroBERT/EuroBERT-610m",
            "EuroBERT/EuroBERT-2.1B",
        ),
        cli=_PromptTuneArgs(
            dataset="multinerd",
            prefix_init="random",
            arch="encoder",
            train_samples=20000,
            val_samples=1024,
        ),
    ),
    _Job(
        job_name="deberta-head-only",
        time="01:00:00",
        mem="16GB",
        cpus=32,
        gres="gpu:h200-141g:1",
        models=(
            "microsoft/deberta-v3-xsmall",
            "microsoft/deberta-v3-small",
            "microsoft/deberta-v3-base",
            "microsoft/deberta-v3-large",
        ),
        cli=_FineTuneArgs(
            dataset="multinerd",
            arch="encoder",
            head_only=True,
            train_samples=20000,
            val_samples=1024,
        ),
    ),
    _Job(
        job_name="deberta-fine-tune",
        time="01:00:00",
        mem="16GB",
        cpus=32,
        gres="gpu:h200-141g:1",
        models=(
            "microsoft/deberta-v3-xsmall",
            "microsoft/deberta-v3-small",
            "microsoft/deberta-v3-base",
            "microsoft/deberta-v3-large",
        ),
        cli=_FineTuneArgs(
            dataset="multinerd",
            arch="encoder",
            head_only=False,
            train_samples=20000,
            val_samples=1024,
        ),
    ),
    _Job(
        job_name="deberta-prompt-tune-pretrained",
        time="01:00:00",
        mem="16GB",
        cpus=32,
        gres="gpu:h200-141g:1",
        models=(
            "microsoft/deberta-v3-xsmall",
            "microsoft/deberta-v3-small",
            "microsoft/deberta-v3-base",
            "microsoft/deberta-v3-large",
        ),
        cli=_PromptTuneArgs(
            dataset="multinerd",
            prefix_init="pretrained",
            arch="encoder",
            train_samples=20000,
            val_samples=1024,
        ),
    ),
    _Job(
        job_name="deberta-prompt-tune-random",
        time="01:00:00",
        mem="16GB",
        cpus=32,
        gres="gpu:h200-141g:1",
        models=(
            "microsoft/deberta-v3-xsmall",
            "microsoft/deberta-v3-small",
            "microsoft/deberta-v3-base",
            "microsoft/deberta-v3-large",
        ),
        cli=_PromptTuneArgs(
            dataset="multinerd",
            prefix_init="random",
            arch="encoder",
            train_samples=20000,
            val_samples=1024,
        ),
    ),
    _Job(
        job_name="gptneox-few-shot",
        time="03:00:00",
        mem="32GB",
        cpus=32,
        gres="gpu:h200-141g:1",
        models=(
            "EleutherAI/pythia-70m",
            "EleutherAI/pythia-160m",
            "EleutherAI/pythia-410m",
            "EleutherAI/pythia-1b",
            "EleutherAI/pythia-1.4b",
            "EleutherAI/pythia-2.8b",
            "EleutherAI/pythia-6.9b",
        ),
        cli=_FewShotArgs(dataset="multinerd", arch="decoder"),
    ),
    _Job(
        job_name="gptneox-fine-tune",
        time="03:00:00",
        mem="32GB",
        cpus=32,
        gres="gpu:h200-141g:1",
        models=(
            "EleutherAI/pythia-70m",
            "EleutherAI/pythia-160m",
            "EleutherAI/pythia-410m",
            "EleutherAI/pythia-1b",
            "EleutherAI/pythia-1.4b",
            "EleutherAI/pythia-2.8b",
            "EleutherAI/pythia-6.9b",
        ),
        cli=_FineTuneArgs(
            dataset="multinerd",
            arch="decoder",
            head_only=False,
            train_samples=20000,
            val_samples=1024,
        ),
    ),
    _Job(
        job_name="gptneox-prompt-tune-pretrained",
        time="03:00:00",
        mem="32GB",
        cpus=32,
        gres="gpu:h200-141g:1",
        models=(
            "EleutherAI/pythia-70m",
            "EleutherAI/pythia-160m",
            "EleutherAI/pythia-410m",
            "EleutherAI/pythia-1b",
            "EleutherAI/pythia-1.4b",
            "EleutherAI/pythia-2.8b",
            "EleutherAI/pythia-6.9b",
        ),
        cli=_PromptTuneArgs(
            dataset="multinerd",
            prefix_init="pretrained",
            arch="decoder",
            train_samples=20000,
            val_samples=1024,
        ),
    ),
    _Job(
        job_name="gptneox-prompt-tune-random",
        time="03:00:00",
        mem="32GB",
        cpus=32,
        gres="gpu:h200-141g:1",
        models=(
            "EleutherAI/pythia-70m",
            "EleutherAI/pythia-160m",
            "EleutherAI/pythia-410m",
            "EleutherAI/pythia-1b",
            "EleutherAI/pythia-1.4b",
            "EleutherAI/pythia-2.8b",
            "EleutherAI/pythia-6.9b",
        ),
        cli=_PromptTuneArgs(
            dataset="multinerd",
            prefix_init="random",
            arch="decoder",
            train_samples=20000,
            val_samples=1024,
        ),
    ),
    _Job(
        job_name="qwen35-few-shot",
        time="04:00:00",
        mem="32GB",
        cpus=32,
        gres="gpu:h200-141g:1",
        models=(
            "Qwen/Qwen3.5-0.8B",
            "Qwen/Qwen3.5-2B",
            "Qwen/Qwen3.5-4B",
            "Qwen/Qwen3.5-9B",
        ),
        cli=_FewShotArgs(dataset="multinerd", arch="decoder"),
    ),
    _Job(
        job_name="qwen35-fine-tune",
        time="04:00:00",
        mem="32GB",
        cpus=32,
        gres="gpu:h200-141g:1",
        models=(
            "Qwen/Qwen3.5-0.8B",
            "Qwen/Qwen3.5-2B",
            "Qwen/Qwen3.5-4B",
            "Qwen/Qwen3.5-9B",
        ),
        cli=_FineTuneArgs(
            dataset="multinerd",
            arch="decoder",
            head_only=False,
            train_samples=20000,
            val_samples=1024,
        ),
    ),
    _Job(
        job_name="qwen35-prompt-tune-pretrained",
        time="04:00:00",
        mem="32GB",
        cpus=32,
        gres="gpu:h200-141g:1",
        models=(
            "Qwen/Qwen3.5-0.8B",
            "Qwen/Qwen3.5-2B",
            "Qwen/Qwen3.5-4B",
            "Qwen/Qwen3.5-9B",
        ),
        cli=_PromptTuneArgs(
            dataset="multinerd",
            prefix_init="pretrained",
            arch="decoder",
            train_samples=20000,
            val_samples=1024,
        ),
    ),
    _Job(
        job_name="qwen35-prompt-tune-random",
        time="04:00:00",
        mem="32GB",
        cpus=32,
        gres="gpu:h200-141g:1",
        models=(
            "Qwen/Qwen3.5-0.8B",
            "Qwen/Qwen3.5-2B",
            "Qwen/Qwen3.5-4B",
            "Qwen/Qwen3.5-9B",
        ),
        cli=_PromptTuneArgs(
            dataset="multinerd",
            prefix_init="random",
            arch="decoder",
            train_samples=20000,
            val_samples=1024,
        ),
    ),
    _Job(
        job_name="llama32-few-shot",
        time="04:00:00",
        mem="32GB",
        cpus=32,
        gres="gpu:h200-141g:1",
        models=(
            "meta-llama/Llama-3.2-1B-Instruct",
            "meta-llama/Llama-3.2-3B-Instruct",
            "meta-llama/Llama-3.1-8B-Instruct",
        ),
        cli=_FewShotArgs(dataset="multinerd", arch="decoder"),
    ),
    _Job(
        job_name="llama32-fine-tune",
        time="04:00:00",
        mem="32GB",
        cpus=32,
        gres="gpu:h200-141g:1",
        models=(
            "meta-llama/Llama-3.2-1B-Instruct",
            "meta-llama/Llama-3.2-3B-Instruct",
            "meta-llama/Llama-3.1-8B-Instruct",
        ),
        cli=_FineTuneArgs(
            dataset="multinerd",
            arch="decoder",
            head_only=False,
            train_samples=20000,
            val_samples=1024,
        ),
    ),
    _Job(
        job_name="llama32-prompt-tune-pretrained",
        time="04:00:00",
        mem="32GB",
        cpus=32,
        gres="gpu:h200-141g:1",
        models=(
            "meta-llama/Llama-3.2-1B-Instruct",
            "meta-llama/Llama-3.2-3B-Instruct",
            "meta-llama/Llama-3.1-8B-Instruct",
        ),
        cli=_PromptTuneArgs(
            dataset="multinerd",
            prefix_init="pretrained",
            arch="decoder",
            train_samples=20000,
            val_samples=1024,
        ),
    ),
    _Job(
        job_name="llama32-prompt-tune-random",
        time="04:00:00",
        mem="32GB",
        cpus=32,
        gres="gpu:h200-141g:1",
        models=(
            "meta-llama/Llama-3.2-1B-Instruct",
            "meta-llama/Llama-3.2-3B-Instruct",
            "meta-llama/Llama-3.1-8B-Instruct",
        ),
        cli=_PromptTuneArgs(
            dataset="multinerd",
            prefix_init="random",
            arch="decoder",
            train_samples=20000,
            val_samples=1024,
        ),
    ),
    _Job(
        job_name="gemma3-few-shot",
        time="04:00:00",
        mem="32GB",
        cpus=32,
        gres="gpu:h200-141g:1",
        models=(
            "google/gemma-3-270m-it",
            "google/gemma-3-1b-it",
            "google/gemma-3-4b-it",
        ),
        cli=_FewShotArgs(dataset="multinerd", arch="decoder"),
    ),
    _Job(
        job_name="gemma3-fine-tune",
        time="04:00:00",
        mem="32GB",
        cpus=32,
        gres="gpu:h200-141g:1",
        models=(
            "google/gemma-3-270m-it",
            "google/gemma-3-1b-it",
            "google/gemma-3-4b-it",
        ),
        cli=_FineTuneArgs(
            dataset="multinerd",
            arch="decoder",
            head_only=False,
            train_samples=20000,
            val_samples=1024,
        ),
    ),
    _Job(
        job_name="gemma3-prompt-tune-pretrained",
        time="04:00:00",
        mem="32GB",
        cpus=32,
        gres="gpu:h200-141g:1",
        models=(
            "google/gemma-3-270m-it",
            "google/gemma-3-1b-it",
            "google/gemma-3-4b-it",
        ),
        cli=_PromptTuneArgs(
            dataset="multinerd",
            prefix_init="pretrained",
            arch="decoder",
            train_samples=20000,
            val_samples=1024,
        ),
    ),
    _Job(
        job_name="gemma3-prompt-tune-random",
        time="04:00:00",
        mem="32GB",
        cpus=32,
        gres="gpu:h200-141g:1",
        models=(
            "google/gemma-3-270m-it",
            "google/gemma-3-1b-it",
            "google/gemma-3-4b-it",
        ),
        cli=_PromptTuneArgs(
            dataset="multinerd",
            prefix_init="random",
            arch="decoder",
            train_samples=20000,
            val_samples=1024,
        ),
    ),
    _Job(
        job_name="flant5-few-shot",
        time="05:00:00",
        mem="32GB",
        cpus=32,
        gres="gpu:h200-141g:1",
        models=(
            "google/flan-t5-small",
            "google/flan-t5-base",
            "google/flan-t5-large",
            "google/flan-t5-xl",
            "google/flan-t5-xxl",
        ),
        cli=_FewShotArgs(dataset="multinerd", arch="encoder-decoder"),
    ),
    _Job(
        job_name="flant5-fine-tune",
        time="05:00:00",
        mem="32GB",
        cpus=32,
        gres="gpu:h200-141g:1",
        models=(
            "google/flan-t5-small",
            "google/flan-t5-base",
            "google/flan-t5-large",
            "google/flan-t5-xl",
            "google/flan-t5-xxl",
        ),
        cli=_FineTuneArgs(
            dataset="multinerd",
            arch="encoder-decoder",
            head_only=False,
            train_samples=20000,
            val_samples=1024,
        ),
    ),
    _Job(
        job_name="flant5-prompt-tune-pretrained",
        time="05:00:00",
        mem="32GB",
        cpus=32,
        gres="gpu:h200-141g:1",
        models=(
            "google/flan-t5-small",
            "google/flan-t5-base",
            "google/flan-t5-large",
            "google/flan-t5-xl",
            "google/flan-t5-xxl",
        ),
        cli=_PromptTuneArgs(
            dataset="multinerd",
            prefix_init="pretrained",
            arch="encoder-decoder",
            train_samples=20000,
            val_samples=1024,
        ),
    ),
    _Job(
        job_name="flant5-prompt-tune-random",
        time="05:00:00",
        mem="32GB",
        cpus=32,
        gres="gpu:h200-141g:1",
        models=(
            "google/flan-t5-small",
            "google/flan-t5-base",
            "google/flan-t5-large",
            "google/flan-t5-xl",
            "google/flan-t5-xxl",
        ),
        cli=_PromptTuneArgs(
            dataset="multinerd",
            prefix_init="random",
            arch="encoder-decoder",
            train_samples=20000,
            val_samples=1024,
        ),
    ),
    _Job(
        job_name="t5gemma-few-shot",
        time="05:00:00",
        mem="32GB",
        cpus=32,
        gres="gpu:h200-141g:1",
        models=(
            "google/t5gemma-2-270m-270m",
            "google/t5gemma-2-1b-1b",
            "google/t5gemma-2-4b-4b",
        ),
        cli=_FewShotArgs(dataset="multinerd", arch="encoder-decoder"),
    ),
    _Job(
        job_name="t5gemma-fine-tune",
        time="05:00:00",
        mem="32GB",
        cpus=32,
        gres="gpu:h200-141g:1",
        models=(
            "google/t5gemma-2-270m-270m",
            "google/t5gemma-2-1b-1b",
            "google/t5gemma-2-4b-4b",
        ),
        cli=_FineTuneArgs(
            dataset="multinerd",
            arch="encoder-decoder",
            head_only=False,
            train_samples=20000,
            val_samples=1024,
        ),
    ),
    _Job(
        job_name="t5gemma-prompt-tune-pretrained",
        time="05:00:00",
        mem="32GB",
        cpus=32,
        gres="gpu:h200-141g:1",
        models=(
            "google/t5gemma-2-270m-270m",
            "google/t5gemma-2-1b-1b",
            "google/t5gemma-2-4b-4b",
        ),
        cli=_PromptTuneArgs(
            dataset="multinerd",
            prefix_init="pretrained",
            arch="encoder-decoder",
            train_samples=20000,
            val_samples=1024,
        ),
    ),
    _Job(
        job_name="t5gemma-prompt-tune-random",
        time="05:00:00",
        mem="32GB",
        cpus=32,
        gres="gpu:h200-141g:1",
        models=(
            "google/t5gemma-2-270m-270m",
            "google/t5gemma-2-1b-1b",
            "google/t5gemma-2-4b-4b",
        ),
        cli=_PromptTuneArgs(
            dataset="multinerd",
            prefix_init="random",
            arch="encoder-decoder",
            train_samples=20000,
            val_samples=1024,
        ),
    ),
]


def _cli(*args: str) -> str:
    return shlex.join(["uv", "run", "--no-sync", "cli", *args])


def _common_args(model: str, cli: _Cli) -> list[str]:
    args = [
        "--model",
        model,
        "--dataset",
        cli.dataset,
        "--n-shot",
        str(cli.n_shot),
        "--batch-size",
        str(cli.batch_size),
        "--mlflow-experiment",
        cli.mlflow_experiment,
        "--log-level",
        cli.log_level,
        "--seed",
        str(cli.seed),
    ]

    if cli.arch is not None:
        args.extend(("--arch", cli.arch))

    if cli.mlflow_run_name is not None:
        args.extend(("--mlflow-run-name", cli.mlflow_run_name))

    if cli.mlflow_tracking_uri is not None:
        args.extend(("--mlflow-tracking-uri", cli.mlflow_tracking_uri))

    return args


def _tuning_args(cli: _FineTuneArgs | _PromptTuneArgs) -> list[str]:
    args = [
        "--epochs",
        str(cli.epochs),
        "--batch-size",
        str(cli.batch_size),
        "--learning-rate",
        str(cli.learning_rate),
    ]

    if cli.train_samples is not None:
        args.extend(("--train-samples", str(cli.train_samples)))

    if cli.val_samples is not None:
        args.extend(("--val-samples", str(cli.val_samples)))

    if cli.do_eval:
        args.append("--do-eval")

    if cli.early_stopping:
        args.append("--early-stopping")

    if not cli.resume:
        args.append("--no-resume")

    return args


def sbatch_args(model: str, job: _Job) -> list[str]:
    """Build a job's sbatch flags for one model, excluding `--wrap`."""
    model_name = model.replace("/", "-")
    args = [
        f"--job-name={job.job_name}",
        f"--time={job.time}",
        f"--mem={job.mem}",
        f"--cpus-per-task={job.cpus}",
        f"--gres={job.gres}",
        f"--partition={job.partition}",
        f"--output={SLURMDIR}/%j[{model_name}]-%x.out",
    ]

    if job.requeue:
        args.append("--requeue")

    if job.signal is not None:
        args.append(f"--signal={job.signal}")

    return args


def command(model: str, job: _Job) -> str:
    """Build a job's single `cli` invocation for one model."""
    cli = job.cli
    args = _common_args(model, cli)

    match cli.command:
        case "few-shot":
            return _cli("few-shot", *args)
        case "fine-tune":
            args += _tuning_args(cli)
            if cli.head_only:
                args.append("--head-only")

            return _cli("fine-tune", *args)
        case "prompt-tune":
            args.extend(_tuning_args(cli))
            args.extend(("--prefix-init", cli.prefix_init))
            return _cli("prompt-tune", *args)
