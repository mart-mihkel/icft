"""Tests for architecture inference, freezing, and trainer construction helpers."""

from typing import TYPE_CHECKING, cast
from unittest.mock import MagicMock, patch

import torch
from torch.nn import Linear
from transformers import BertConfig, GPT2Config, Seq2SeqTrainingArguments, T5Config
from transformers.trainer import Trainer

from saspbft.constants import LOGDIR
from saspbft.modeling import (
    Gemma3Trainer,
    StripTokenTypeIds,
    _patch_gemma3,
    freeze,
    get_arch,
    get_args,
    get_n_virtual,
    save_model,
)

if TYPE_CHECKING:
    from pathlib import Path

    from transformers import PreTrainedTokenizerFast


def test_get_arch_override_short_circuits() -> None:
    assert get_arch(T5Config(), override="decoder") == "decoder"


def test_get_arch_encoder_decoder() -> None:
    assert get_arch(T5Config()) == "encoder-decoder"


def test_get_arch_encoder() -> None:
    assert get_arch(BertConfig()) == "encoder"


def test_get_arch_decoder() -> None:
    assert get_arch(GPT2Config()) == "decoder"


def test_freeze_default_freezes_all_params() -> None:
    model = Linear(4, 2)
    freeze(model)

    assert all(not p.requires_grad for p in model.parameters())


def test_freeze_keeps_skipped_params_trainable() -> None:
    model = Linear(4, 2)
    freeze(model, skip={"weight"})

    assert model.weight.requires_grad
    assert not model.bias.requires_grad


def test_get_n_virtual_without_chat_template(
    bert_tokenizer: PreTrainedTokenizerFast,
) -> None:
    n = get_n_virtual(bert_tokenizer, "hello there")
    expected = len(bert_tokenizer("hello there")["input_ids"])

    assert n == expected


def test_get_n_virtual_with_chat_template(
    llama_tokenizer: PreTrainedTokenizerFast,
) -> None:
    n = get_n_virtual(llama_tokenizer, "hello there")
    conv = [{"role": "system", "content": "hello there"}]
    sys_enc = cast(
        "dict[str, list[int]]",
        llama_tokenizer.apply_chat_template(conv),
    )
    expected = len(sys_enc["input_ids"])

    assert n == expected


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
