"""Tests for trainer construction, training arguments, and Gemma 3 quirks."""

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import torch
from transformers import Seq2SeqTrainingArguments
from transformers.trainer import Trainer

from saspbft.constants import LOGDIR
from saspbft.modeling.trainer import (
    Gemma3Trainer,
    StripTokenTypeIds,
    _patch_gemma3,
    find_checkpoint,
    get_args,
    save_model,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_strip_token_type_ids_removes_key() -> None:
    collator = MagicMock(return_value={"input_ids": [1], "token_type_ids": [0]})
    wrapped = StripTokenTypeIds(collator)

    batch = wrapped([{"input_ids": [1]}])

    assert "token_type_ids" not in batch
    assert batch["input_ids"] == [1]


def test_save_model_uses_trainer_output_dir(tmp_path: Path) -> None:
    model = MagicMock()
    trainer = MagicMock()
    trainer.args.output_dir = str(tmp_path / "out")

    save_model(model, trainer, run_name="run")

    model.save_pretrained.assert_called_once_with(str(tmp_path / "out"))


def test_save_model_falls_back_to_run_name() -> None:
    model = MagicMock()
    trainer = MagicMock()
    trainer.args.output_dir = None

    save_model(model, trainer, run_name="my-run")

    model.save_pretrained.assert_called_once_with(str(LOGDIR / "my-run"))


def test_find_checkpoint_without_output_dir(tmp_path: Path) -> None:
    with patch("saspbft.modeling.trainer.LOGDIR", tmp_path):
        assert find_checkpoint("run") is None


def test_find_checkpoint_without_checkpoints(tmp_path: Path) -> None:
    (tmp_path / "run").mkdir()

    with patch("saspbft.modeling.trainer.LOGDIR", tmp_path):
        assert find_checkpoint("run") is None


def test_find_checkpoint_returns_latest(tmp_path: Path) -> None:
    (tmp_path / "run" / "checkpoint-10").mkdir(parents=True)
    (tmp_path / "run" / "checkpoint-200").mkdir(parents=True)

    with patch("saspbft.modeling.trainer.LOGDIR", tmp_path):
        checkpoint = find_checkpoint("run")

    assert checkpoint == str(tmp_path / "run" / "checkpoint-200")


def test_get_args_without_checkpoints_does_not_save() -> None:
    args = get_args("decoder", do_eval=False)

    assert args.save_strategy == "no"


def test_get_args_with_checkpoints_saves_each_epoch() -> None:
    args = get_args("decoder", do_eval=False, checkpoints=True)

    assert args.save_strategy == "epoch"
    assert args.save_total_limit == 1


def test_get_args_encoder_decoder_with_checkpoints_saves_each_epoch() -> None:
    args = get_args("encoder-decoder", do_eval=False, checkpoints=True)

    assert isinstance(args, Seq2SeqTrainingArguments)
    assert args.save_strategy == "epoch"
    assert args.save_total_limit == 1


def test_get_args_encoder_decoder_uses_seq2seq_args() -> None:
    args = get_args("encoder-decoder", do_eval=True)

    assert isinstance(args, Seq2SeqTrainingArguments)
    assert args.predict_with_generate is True
    assert args.eval_strategy == "epoch"


def test_get_args_decoder_uses_plain_args() -> None:
    args = get_args("decoder", do_eval=False)

    assert not isinstance(args, Seq2SeqTrainingArguments)
    assert args.eval_strategy == "no"


def test_get_args_without_cuda_uses_cpu_optim() -> None:
    with patch("torch.cuda.is_available", return_value=False):
        args = get_args("decoder", do_eval=False)

    assert args.optim == "adamw_torch_fused"
    assert args.use_cpu is True
    assert args.bf16 is False


def test_gemma3_trainer_defaults_ignore_keys_to_past_key_values() -> None:
    trainer = Gemma3Trainer.__new__(Gemma3Trainer)
    captured = {}

    def fake_prediction_step(
        self: Trainer,
        model: object,
        inputs: dict,
        prediction_loss_only: bool,
        ignore_keys: list[str] | None = None,
    ) -> tuple[None, None, None]:
        del self, model, inputs, prediction_loss_only
        captured["ignore_keys"] = ignore_keys
        return None, None, None

    with patch.object(Trainer, "prediction_step", fake_prediction_step):
        trainer.prediction_step(MagicMock(), {}, False)

    assert captured["ignore_keys"] == ["past_key_values"]


def test_gemma3_trainer_appends_past_key_values_once() -> None:
    trainer = Gemma3Trainer.__new__(Gemma3Trainer)
    captured = {}

    def fake_prediction_step(
        self: Trainer,
        model: object,
        inputs: dict,
        prediction_loss_only: bool,
        ignore_keys: list[str] | None = None,
    ) -> tuple[None, None, None]:
        del self, model, inputs, prediction_loss_only
        captured["ignore_keys"] = ignore_keys
        return None, None, None

    with patch.object(Trainer, "prediction_step", fake_prediction_step):
        trainer.prediction_step(MagicMock(), {}, False, ignore_keys=["foo"])

    assert captured["ignore_keys"] == ["foo", "past_key_values"]


def test_gemma3_trainer_does_not_duplicate_past_key_values() -> None:
    trainer = Gemma3Trainer.__new__(Gemma3Trainer)
    captured = {}

    def fake_prediction_step(
        self: Trainer,
        model: object,
        inputs: dict,
        prediction_loss_only: bool,
        ignore_keys: list[str] | None = None,
    ) -> tuple[None, None, None]:
        del self, model, inputs, prediction_loss_only
        captured["ignore_keys"] = ignore_keys
        return None, None, None

    with patch.object(Trainer, "prediction_step", fake_prediction_step):
        trainer.prediction_step(
            MagicMock(),
            {},
            False,
            ignore_keys=["foo", "past_key_values"],
        )

    assert captured["ignore_keys"] == ["foo", "past_key_values"]


def test_patch_gemma3_injects_token_type_ids() -> None:
    captured = {}

    def base_forward(*args: object, **kwargs: object) -> str:
        del args
        captured.update(kwargs)
        return "output"

    base_model = MagicMock()
    base_model.forward = base_forward
    model = MagicMock()
    model.base_model.model = base_model

    _patch_gemma3(model)

    attn = torch.tensor([[1, 1, 0]])
    result = base_model.forward(attention_mask=attn)

    assert result == "output"
    assert torch.equal(captured["token_type_ids"], torch.zeros_like(attn))
