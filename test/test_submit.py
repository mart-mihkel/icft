"""Tests for the Slurm job submission error handling."""

import pytest

from saspbft.scripts.submit import submit


@pytest.fixture(autouse=True)
def _sbatch_available(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/sbatch")


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


def test_submit_duplicate_job_name_exits(monkeypatch: pytest.MonkeyPatch) -> None:
    from saspbft.slurm import JOBS

    monkeypatch.setattr("saspbft.scripts.submit.JOBS", [JOBS[0], JOBS[0]])

    with pytest.raises(SystemExit) as exc_info:
        submit(JOBS[0].job_name)

    assert exc_info.value.code == 1
