"""Tests for the Slurm job submission error handling."""

from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

from saspbft.scripts.submit import submit

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(autouse=True)
def _sbatch_available(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/sbatch")


@pytest.fixture(autouse=True)
def _slurm_logdir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr("saspbft.scripts.submit.SLURMDIR", tmp_path / "slurm")


def test_submit_missing_sbatch_exits(monkeypatch: pytest.MonkeyPatch) -> None:
    from saspbft.slurm import JOBS

    monkeypatch.setattr("shutil.which", lambda _: None)

    with pytest.raises(SystemExit) as exc_info:
        submit(JOBS[0].job_name)

    assert exc_info.value.code == 1


def test_submit_unknown_job_exits_without_traceback() -> None:
    with pytest.raises(SystemExit) as exc_info:
        submit("bogus-job")

    assert exc_info.value.code == 1


def test_submit_regex_job_name_matches(monkeypatch: pytest.MonkeyPatch) -> None:
    run_mock = MagicMock()
    monkeypatch.setattr("saspbft.scripts.submit.subprocess.run", run_mock)

    submit("llama.*")

    num_matches = 12
    assert run_mock.call_count == num_matches


def test_submit_passes_sbatch_flags_and_wrap(monkeypatch: pytest.MonkeyPatch) -> None:
    run_mock = MagicMock()
    monkeypatch.setattr("saspbft.scripts.submit.subprocess.run", run_mock)

    submit("llama32-few-shot")

    cmd = run_mock.call_args.args[0]
    assert cmd[0] == "sbatch"
    assert "--job-name=llama32-few-shot" in cmd
    assert "--partition=gpu" in cmd
    assert cmd[-1].startswith("--wrap=uv run --no-sync cli few-shot")


def test_submit_creates_slurm_logdir(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr("saspbft.scripts.submit.subprocess.run", MagicMock())

    submit("llama32-few-shot")

    assert (tmp_path / "slurm").is_dir()
