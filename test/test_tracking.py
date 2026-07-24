"""Tests for exporting MLflow experiment runs to a dataframe/csv."""

from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

from saspbft.scripts.tracking import collect_metrics

if TYPE_CHECKING:
    from pathlib import Path


def _fake_run(run_id: str, metrics: dict, params: dict) -> MagicMock:
    run = MagicMock()
    run.info.run_id = run_id
    run.info.run_name = f"run-{run_id}"
    run.info.status = "FINISHED"
    run.info.start_time = 0
    run.info.end_time = 1
    run.data.metrics = metrics
    run.data.params = params
    return run


def test_collect_metrics_raises_when_experiment_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = MagicMock()
    client.get_experiment_by_name.return_value = None
    monkeypatch.setattr(
        "saspbft.scripts.tracking.MlflowClient",
        lambda *_, **__: client,
    )

    with pytest.raises(RuntimeError, match="missing-experiment"):
        collect_metrics("missing-experiment", "sqlite:///unused.db")


def test_collect_metrics_builds_dataframe_and_writes_csv(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    client = MagicMock()
    experiment = MagicMock(experiment_id="1")
    client.get_experiment_by_name.return_value = experiment
    client.search_runs.return_value = [
        _fake_run("a", {"accuracy": 0.9}, {"dataset": "boolq"}),
        _fake_run("b", {"accuracy": 0.8}, {"dataset": "wic"}),
    ]
    monkeypatch.setattr(
        "saspbft.scripts.tracking.MlflowClient",
        lambda *_, **__: client,
    )
    monkeypatch.setattr("saspbft.scripts.tracking.LOGDIR", tmp_path)

    df = collect_metrics("exp", "sqlite:///unused.db", write_csv=True)
    expected_rows = 2

    assert df.shape[0] == expected_rows
    assert set(df["run_id"]) == {"a", "b"}
    assert set(df["accuracy"]) == {0.9, 0.8}
    assert (tmp_path / "metrics" / "exp.csv").exists()


def test_collect_metrics_skips_csv_by_default(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    client = MagicMock()
    experiment = MagicMock(experiment_id="1")
    client.get_experiment_by_name.return_value = experiment
    client.search_runs.return_value = [_fake_run("a", {"accuracy": 0.9}, {})]
    monkeypatch.setattr(
        "saspbft.scripts.tracking.MlflowClient",
        lambda *_, **__: client,
    )
    monkeypatch.setattr("saspbft.scripts.tracking.LOGDIR", tmp_path)

    collect_metrics("exp", "sqlite:///unused.db")

    assert not (tmp_path / "metrics" / "exp.csv").exists()
