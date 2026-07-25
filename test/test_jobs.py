"""Tests for predefined Slurm job definitions and their cli command output."""

from saspbft.slurm import JOBS, _Cli, _Job, command

_MODEL = "m"

_JOB = _Job(
    job_name="j",
    time="00:10:00",
    mem="1GB",
    cpus=1,
    gres="gpu:1",
    models=(_MODEL,),
    cli=_Cli(
        command="few-shot",
        dataset="multinerd",
        arch="decoder",
        prefix_init="pretrained",
        head_only=False,
        train_samples=10,
        val_samples=5,
        epochs=1,
        batch_size=2,
        experiment="test",
        log_level="debug",
        seed=0,
    ),
)


def test_command_few_shot_has_no_training_args() -> None:
    job = _JOB._replace(cli=_JOB.cli._replace(command="few-shot"))

    cmd = command(_MODEL, job)

    assert "few-shot" in cmd
    assert "--train-samples" not in cmd
    assert "--head-only" not in cmd
    assert "--prefix-init" not in cmd


def test_command_fine_tune_includes_training_args() -> None:
    job = _JOB._replace(cli=_JOB.cli._replace(command="fine-tune", head_only=False))

    cmd = command(_MODEL, job)

    assert "fine-tune" in cmd
    assert "--train-samples 10" in cmd
    assert "--head-only" not in cmd


def test_command_fine_tune_head_only_adds_flag() -> None:
    job = _JOB._replace(cli=_JOB.cli._replace(command="fine-tune", head_only=True))

    cmd = command(_MODEL, job)

    assert "--head-only" in cmd


def test_command_prompt_tune_includes_prefix_init() -> None:
    job = _JOB._replace(
        cli=_JOB.cli._replace(command="prompt-tune", prefix_init="random")
    )

    cmd = command(_MODEL, job)

    assert "prompt-tune" in cmd
    assert "--prefix-init random" in cmd


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
        assert any(job.cli.command == "fine-tune" and job.cli.head_only for job in jobs)
