"""Tests for exporting MLflow experiment runs to a dataframe/csv."""

from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

from saspbft.scripts.tracking import collect_metrics, run_name, start_run

if TYPE_CHECKING:
    from pathlib import Path


def _fake_run(
    run_id: str,
    metrics: dict,
    params: dict,
    status: str = "FINISHED",
) -> MagicMock:
    run = MagicMock()
    run.info.run_id = run_id
    run.info.run_name = f"run-{run_id}"
    run.info.status = status
    run.info.start_time = 0
    run.info.end_time = 1
    run.data.metrics = metrics
    run.data.params = params
    return run


def test_run_name_strips_model_org() -> None:
    name = run_name("multinerd", "meta-llama/Llama-3.1-8B-Instruct", "fine-tune", 20000)

    assert name == "multinerd/Llama-3.1-8B-Instruct/fine-tune/20000"


def test_run_name_without_org() -> None:
    name = run_name("multinerd", "some-model", "fine-tune", 20000)

    assert name == "multinerd/some-model/fine-tune/20000"


def test_run_name_all_samples() -> None:
    name = run_name("obl", "google/flan-t5-xxl", "prompt-tune-random", None)

    assert name == "obl/flan-t5-xxl/prompt-tune-random/all"


def test_run_name_zero_samples_is_not_all() -> None:
    name = run_name("obl", "google/flan-t5-xxl", "5-shot", 0)

    assert name == "obl/flan-t5-xxl/5-shot/0"


def test_start_run_without_resume_starts_new_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = MagicMock()
    monkeypatch.setattr("saspbft.scripts.tracking.mlflow", fake)

    start_run("exp", "my-run", "sqlite:///unused.db", resume=False)

    fake.set_tracking_uri.assert_called_once_with("sqlite:///unused.db")
    fake.set_experiment.assert_called_once_with("exp")
    fake.search_runs.assert_not_called()
    fake.start_run.assert_called_once_with(run_name="my-run")


def test_start_run_reattaches_to_previous_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = MagicMock()
    fake.search_runs.return_value = [_fake_run("a", {}, {}, status="RUNNING")]
    monkeypatch.setattr("saspbft.scripts.tracking.mlflow", fake)

    start_run("exp", "my-run", "sqlite:///unused.db", resume=True)

    fake.start_run.assert_called_once_with(run_id="a")


def test_start_run_does_not_reattach_to_finished_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = MagicMock()
    fake.search_runs.return_value = [_fake_run("a", {}, {})]
    monkeypatch.setattr("saspbft.scripts.tracking.mlflow", fake)

    start_run("exp", "my-run", "sqlite:///unused.db", resume=True)

    fake.start_run.assert_called_once_with(run_name="my-run")


def test_start_run_resume_without_previous_run_starts_new_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = MagicMock()
    fake.search_runs.return_value = []
    monkeypatch.setattr("saspbft.scripts.tracking.mlflow", fake)

    start_run("exp", "my-run", "sqlite:///unused.db", resume=True)

    fake.start_run.assert_called_once_with(run_name="my-run")


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


def test_collect_metrics_sums_train_runtime_over_resumes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    client = MagicMock()
    client.get_experiment_by_name.return_value = MagicMock(experiment_id="1")
    client.search_runs.return_value = [_fake_run("a", {"train_runtime": 30.0}, {})]
    client.get_metric_history.return_value = [
        MagicMock(value=100.0),
        MagicMock(value=30.0),
    ]
    monkeypatch.setattr(
        "saspbft.scripts.tracking.MlflowClient",
        lambda *_, **__: client,
    )
    monkeypatch.setattr("saspbft.scripts.tracking.LOGDIR", tmp_path)

    df = collect_metrics("exp", "sqlite:///unused.db")

    client.get_metric_history.assert_called_once_with("a", "train_runtime")
    assert df["train_runtime"].to_list() == [130.0]


def test_collect_metrics_skips_history_without_train_runtime(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    client = MagicMock()
    client.get_experiment_by_name.return_value = MagicMock(experiment_id="1")
    client.search_runs.return_value = [_fake_run("a", {"test_runtime": 5.0}, {})]
    monkeypatch.setattr(
        "saspbft.scripts.tracking.MlflowClient",
        lambda *_, **__: client,
    )
    monkeypatch.setattr("saspbft.scripts.tracking.LOGDIR", tmp_path)

    collect_metrics("exp", "sqlite:///unused.db")

    client.get_metric_history.assert_not_called()


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
