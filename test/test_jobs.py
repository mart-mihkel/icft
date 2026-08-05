"""Tests for predefined Slurm job definitions and their cli command output."""

from typing import TYPE_CHECKING

from saspbft.constants import SLURMDIR
from saspbft.scripts.cli import few_shot, fine_tune, prompt_tune
from saspbft.slurm import (
    JOBS,
    _FewShotArgs,
    _FineTuneArgs,
    _Job,
    _PromptTuneArgs,
    command,
    sbatch_args,
)

if TYPE_CHECKING:
    from click import Command

    from saspbft.types import PrefixInit


type _Default = str | int | float | bool | tuple[str, ...] | None


def _click_option_names(cmd: Command) -> set[str]:
    """Names of a click command's options, excluding `--model` (from `_Job.models`)."""
    return {p.name for p in cmd.params if p.name is not None} - {"model"}


def _mismatched_defaults(
    cmd: Command,
    field_defaults: dict[str, _Default],
) -> dict[str, tuple[_Default, _Default]]:
    """Map each field whose job default drifted from the cli default to both."""
    click_defaults = {
        p.name: cast("_Default", p.default) for p in cmd.params if p.name is not None
    }

    return {
        name: (default, click_defaults[name])
        for name, default in field_defaults.items()
        if name in click_defaults and click_defaults[name] != default
    }


def test_fine_tune_args_matches_cli_options() -> None:
    assert _click_option_names(fine_tune) == set(_FineTuneArgs._fields) - {"command"}


def test_prompt_tune_args_matches_cli_options() -> None:
    assert _click_option_names(prompt_tune) == set(_PromptTuneArgs._fields) - {
        "command"
    }


def test_few_shot_args_matches_cli_options() -> None:
    assert _click_option_names(few_shot) == set(_FewShotArgs._fields) - {"command"}


def test_fine_tune_args_defaults_match_cli_options() -> None:
    assert _mismatched_defaults(fine_tune, _FineTuneArgs._field_defaults) == {}


def test_prompt_tune_args_defaults_match_cli_options() -> None:
    assert _mismatched_defaults(prompt_tune, _PromptTuneArgs._field_defaults) == {}


def test_few_shot_args_defaults_match_cli_options() -> None:
    assert _mismatched_defaults(few_shot, _FewShotArgs._field_defaults) == {}


def _job(
    cli: _FewShotArgs | _FineTuneArgs | _PromptTuneArgs,
    *,
    partition: str = "gpu",
    requeue: bool = True,
    signal: str | None = None,
) -> _Job:
    return _Job(
        job_name="j",
        time="00:10:00",
        mem="1GB",
        cpus=1,
        gres="gpu:1",
        models=(_MODEL,),
        cli=cli,
        partition=partition,
        requeue=requeue,
        signal=signal,
    )


def _few_shot_args(
    *,
    n_shot: int = 5,
    mlflow_run_name: str | None = None,
    mlflow_tracking_uri: str | None = None,
) -> _FewShotArgs:
    return _FewShotArgs(
        dataset="multinerd",
        arch="decoder",
        batch_size=2,
        mlflow_experiment="test",
        log_level="debug",
        seed=0,
        n_shot=n_shot,
        mlflow_run_name=mlflow_run_name,
        mlflow_tracking_uri=mlflow_tracking_uri,
    )


def _fine_tune_args(
    *,
    head_only: bool = False,
    do_eval: bool = False,
    early_stopping: bool = False,
    resume: bool = True,
    learning_rate: float = 5e-5,
    mlflow_run_name: str | None = None,
    mlflow_tracking_uri: str | None = None,
) -> _FineTuneArgs:
    return _FineTuneArgs(
        dataset="multinerd",
        arch="decoder",
        head_only=head_only,
        train_samples=10,
        val_samples=5,
        epochs=1,
        batch_size=2,
        mlflow_experiment="test",
        log_level="debug",
        seed=0,
        do_eval=do_eval,
        early_stopping=early_stopping,
        resume=resume,
        learning_rate=learning_rate,
        mlflow_run_name=mlflow_run_name,
        mlflow_tracking_uri=mlflow_tracking_uri,
    )


def _prompt_tune_args(*, prefix_init: PrefixInit) -> _PromptTuneArgs:
    return _PromptTuneArgs(
        dataset="multinerd",
        prefix_init=prefix_init,
        arch="decoder",
        train_samples=10,
        val_samples=5,
        epochs=1,
        batch_size=2,
        mlflow_experiment="test",
        log_level="debug",
        seed=0,
    )


def test_command_few_shot_has_no_training_args() -> None:
    job = _job(_few_shot_args())

    cmd = command(_MODEL, job)

    assert "few-shot" in cmd
    assert "--train-samples" not in cmd
    assert "--head-only" not in cmd
    assert "--prefix-init" not in cmd


def test_command_few_shot_includes_n_shot() -> None:
    job = _job(_few_shot_args(n_shot=7))

    cmd = command(_MODEL, job)

    assert "--n-shot 7" in cmd


def test_command_fine_tune_includes_training_args() -> None:
    job = _job(_fine_tune_args(head_only=False))

    cmd = command(_MODEL, job)

    assert "fine-tune" in cmd
    assert "--train-samples 10" in cmd
    assert "--head-only" not in cmd


def test_command_fine_tune_head_only_adds_flag() -> None:
    job = _job(_fine_tune_args(head_only=True))

    cmd = command(_MODEL, job)

    assert "--head-only" in cmd


def test_command_fine_tune_resumes_by_default() -> None:
    job = _job(_fine_tune_args())

    cmd = command(_MODEL, job)

    assert "--no-resume" not in cmd


def test_command_fine_tune_no_resume_adds_flag() -> None:
    job = _job(_fine_tune_args(resume=False))

    cmd = command(_MODEL, job)

    assert "--no-resume" in cmd


def test_command_prompt_tune_includes_prefix_init() -> None:
    job = _job(_prompt_tune_args(prefix_init="random"))

    cmd = command(_MODEL, job)

    assert "prompt-tune" in cmd
    assert "--prefix-init random" in cmd


def test_command_includes_optional_tracking_args_when_set() -> None:
    job = _job(
        _fine_tune_args(
            do_eval=True,
            early_stopping=True,
            learning_rate=1e-4,
            mlflow_run_name="run",
            mlflow_tracking_uri="sqlite:///other.db",
        )
    )

    cmd = command(_MODEL, job)

    assert "--do-eval" in cmd
    assert "--early-stopping" in cmd
    assert "--learning-rate 0.0001" in cmd
    assert "--mlflow-run-name run" in cmd
    assert "--mlflow-tracking-uri sqlite:///other.db" in cmd


def test_command_omits_optional_tracking_args_when_unset() -> None:
    job = _job(_fine_tune_args())

    cmd = command(_MODEL, job)

    assert "--do-eval" not in cmd
    assert "--early-stopping" not in cmd
    assert "--mlflow-run-name" not in cmd
    assert "--mlflow-tracking-uri" not in cmd


def test_sbatch_args_includes_resource_flags() -> None:
    job = _job(_few_shot_args())

    args = sbatch_args(_MODEL, job)

    assert "--job-name=j" in args
    assert "--time=00:10:00" in args
    assert "--mem=1GB" in args
    assert "--cpus-per-task=1" in args
    assert "--gres=gpu:1" in args
    assert "--partition=gpu" in args


def test_sbatch_args_uses_job_partition() -> None:
    job = _job(_few_shot_args(), partition="main")

    args = sbatch_args(_MODEL, job)

    assert "--partition=main" in args
    assert "--partition=gpu" not in args


def test_sbatch_args_escapes_model_name_in_output() -> None:
    job = _job(_few_shot_args())

    args = sbatch_args("org/some-model", job)

    assert f"--output={SLURMDIR}/%j[org-some-model]-%x.out" in args


def test_sbatch_args_requeues_by_default() -> None:
    job = _job(_few_shot_args())

    args = sbatch_args(_MODEL, job)

    assert "--requeue" in args


def test_sbatch_args_omits_requeue_when_disabled() -> None:
    job = _job(_few_shot_args(), requeue=False)

    args = sbatch_args(_MODEL, job)

    assert "--requeue" not in args


def test_sbatch_args_omits_signal_by_default() -> None:
    job = _job(_few_shot_args())

    args = sbatch_args(_MODEL, job)

    assert not any(arg.startswith("--signal") for arg in args)


def test_sbatch_args_includes_signal_when_set() -> None:
    job = _job(_few_shot_args(), signal="USR1@300")

    args = sbatch_args(_MODEL, job)

    assert "--signal=USR1@300" in args


def test_sbatch_args_omits_wrap() -> None:
    job = _job(_few_shot_args())

    args = sbatch_args(_MODEL, job)

    assert not any(arg.startswith("--wrap") for arg in args)


def test_jobs_registry_job_names_are_unique() -> None:
    names = [job.job_name for job in JOBS]
    assert len(names) == len(set(names))


def test_jobs_registry_all_models_are_nonempty_strings() -> None:
    for job in JOBS:
        assert job.models
        for model in job.models:
            assert isinstance(model, str)
            assert model


def test_jobs_registry_head_only_families_have_head_only_fine_tune() -> None:
    for prefix in ("distilbert", "mmbert", "eurobert", "deberta"):
        jobs = [job for job in JOBS if job.job_name.startswith(prefix)]
        assert jobs
        assert any(
            isinstance(job.cli, _FineTuneArgs) and job.cli.head_only for job in jobs
        )
