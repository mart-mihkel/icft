"""Tests for CLI helper functions and command wiring."""

import random
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import numpy
import torch
from click.testing import CliRunner

from saspbft.scripts.cli import _set_seed, app

if TYPE_CHECKING:
    import pytest

runner = CliRunner()


def test_set_seed_is_deterministic() -> None:
    _set_seed(0)
    a = (random.random(), numpy.random.rand(), torch.rand(1).item())  # noqa: NPY002, S311

    _set_seed(0)
    b = (random.random(), numpy.random.rand(), torch.rand(1).item())  # noqa: NPY002, S311

    assert a == b


def test_set_seed_different_seeds_differ() -> None:
    _set_seed(0)
    a = torch.rand(1).item()

    _set_seed(1)
    b = torch.rand(1).item()

    assert a != b


def test_fine_tune_command_forwards_args(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = MagicMock()
    monkeypatch.setattr("saspbft.scripts.fine.fine_tune", fake)
    epochs = 2

    result = runner.invoke(
        app,
        [
            "fine-tune",
            "--model",
            "my-model",
            "--dataset",
            "boolq",
            "--epochs",
            str(epochs),
            "--head-only",
        ],
    )

    assert result.exit_code == 0, result.output
    fake.assert_called_once()
    kwargs = fake.call_args.kwargs
    assert kwargs["model_path"] == "my-model"
    assert kwargs["dataset"] == "boolq"
    assert kwargs["epochs"] == epochs
    assert kwargs["head_only"] is True


def test_fine_tune_command_sets_seed_when_provided(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("saspbft.scripts.fine.fine_tune", MagicMock())
    fake_set_seed = MagicMock()
    monkeypatch.setattr("saspbft.scripts.cli._set_seed", fake_set_seed)

    result = runner.invoke(
        app,
        ["fine-tune", "--model", "my-model", "--dataset", "boolq", "--seed", "7"],
    )

    assert result.exit_code == 0, result.output
    fake_set_seed.assert_called_once_with(7)


def test_prompt_tune_command_forwards_args(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = MagicMock()
    monkeypatch.setattr("saspbft.scripts.prompt.prompt_tune", fake)

    result = runner.invoke(
        app,
        [
            "prompt-tune",
            "--model",
            "my-model",
            "--dataset",
            "wic",
            "--prefix-init",
            "random",
        ],
    )

    assert result.exit_code == 0, result.output
    fake.assert_called_once()
    kwargs = fake.call_args.kwargs
    assert kwargs["model_path"] == "my-model"
    assert kwargs["dataset"] == "wic"
    assert kwargs["prefix_init"] == "random"


def test_few_shot_command_forwards_args(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = MagicMock()
    monkeypatch.setattr("saspbft.scripts.fewshot.few_shot", fake)
    n_shot = 3

    result = runner.invoke(
        app,
        [
            "few-shot",
            "--model",
            "my-model",
            "--dataset",
            "obl",
            "--n-shot",
            str(n_shot),
        ],
    )

    assert result.exit_code == 0, result.output
    fake.assert_called_once()
    kwargs = fake.call_args.kwargs
    assert kwargs["model_path"] == "my-model"
    assert kwargs["dataset"] == "obl"
    assert kwargs["n_shot"] == n_shot


def test_submit_command_forwards_job(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = MagicMock()
    monkeypatch.setattr("saspbft.scripts.submit.submit", fake)

    result = runner.invoke(app, ["submit", "--job", "llama32-fewshot"])

    assert result.exit_code == 0, result.output
    fake.assert_called_once_with("llama32-fewshot")


def test_collect_metrics_command_forwards_args(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = MagicMock()
    monkeypatch.setattr("saspbft.scripts.tracking.collect_metrics", fake)

    result = runner.invoke(
        app,
        ["collect", "--mlflow-experiment", "my-exp"],
    )

    assert result.exit_code == 0, result.output
    fake.assert_called_once_with("my-exp", "sqlite:///mlflow.db", write_csv=True)
