"""Trainer construction, training arguments, and Gemma 3 / seq2seq quirks."""

from typing import TYPE_CHECKING, Any, cast

import torch
from peft import PeftModel
from transformers import (
    DataCollator,
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments

from saspbft.constants import LOGDIR
from saspbft.logging import logger

if TYPE_CHECKING:
    from collections.abc import Callable

    from datasets.dataset_dict import DatasetDict
    from torch import Tensor
    from torch.nn import Module
    from torch.utils.data import Dataset
    from transformers import EvalPrediction, PreTrainedModel
    from transformers.models.gemma3.modeling_gemma3 import Gemma3ModelOutputWithPast

    from saspbft.types import Architecture


class Gemma3Trainer(Trainer):
    """Trainer that drops the kv-cache during prediction for multimodal Gemma 3."""

    def prediction_step(
        self,
        model: Module,
        inputs: dict[str, Tensor | Any],
        prediction_loss_only: bool,
        ignore_keys: list[str] | None = None,
    ) -> tuple[Tensor | None, Tensor | None, Tensor | None]:
        """
        Ignore kv-cache.

        Multimodal Gemma 3 errors out using DynamicCache.
        """
        if ignore_keys is None:
            ignore_keys = ["past_key_values"]
        elif "past_key_values" not in ignore_keys:
            ignore_keys.append("past_key_values")

        return super().prediction_step(
            model,
            inputs,
            prediction_loss_only,
            ignore_keys=ignore_keys,
        )


class StripTokenTypeIds:
    """Wrap a collator and drop `token_type_ids`, unsupported by seq2seq models."""

    def __init__(self, collator: DataCollator) -> None:
        """Store the collator to wrap."""
        self.collator = collator

    def __call__(self, features: list[dict]) -> dict:
        """Collate features and strip `token_type_ids` from the batch."""
        batch = self.collator(features)
        batch.pop("token_type_ids", None)
        return batch


def _patch_gemma3(model: PeftModel) -> None:
    """
    Patch Gemma 3 fowrad pass for pormpt tuning.

    Multimodal Gemma 3 needs token type ids but peft prompt tuning drops them.
    """
    _base_model = model.base_model.model
    _original_forward = _base_model.forward

    def _gemma3_patched_forward(
        *args: Any,  # noqa: ANN401
        **kwargs: Any,  # noqa: ANN401
    ) -> tuple | Gemma3ModelOutputWithPast:
        ref = kwargs["attention_mask"]
        kwargs["token_type_ids"] = torch.zeros_like(ref)
        return _original_forward(*args, **kwargs)

    _base_model.forward = _gemma3_patched_forward


def get_args(
    arch: Architecture,
    do_eval: bool,
    *,
    epochs: int = 0,
    learning_rate: float = 5e-5,
    batch_size: int = 8,
    run_name: str = "default",
    report_to: str = "none",
) -> TrainingArguments:
    """Build Trainer arguments, using seq2seq args for encoder-decoder models."""
    have_cuda = torch.cuda.is_available()
    optim = "adamw_8bit" if have_cuda else "adamw_torch_fused"
    eval_strategy = "epoch" if do_eval else "no"
    out_dir = LOGDIR / run_name

    if arch == "encoder-decoder":
        logger.debug("use seq2seq training args")
        args = Seq2SeqTrainingArguments(
            run_name=run_name,
            report_to=report_to,
            output_dir=str(out_dir),
            save_strategy="no",
            eval_strategy=eval_strategy,
            eval_on_start=do_eval,
            batch_eval_metrics=True,
            remove_unused_columns=False,
            logging_steps=100,
            metric_for_best_model="f1",
            learning_rate=learning_rate,
            optim=optim,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            use_cpu=not have_cuda,
            bf16_full_eval=have_cuda,
            bf16=have_cuda,
            predict_with_generate=True,
            generation_max_length=8,
        )
    else:
        logger.debug("use regular training args")
        args = TrainingArguments(
            run_name=run_name,
            report_to=report_to,
            output_dir=str(out_dir),
            save_strategy="no",
            eval_strategy=eval_strategy,
            eval_on_start=do_eval,
            batch_eval_metrics=True,
            remove_unused_columns=False,
            logging_steps=100,
            metric_for_best_model="f1",
            learning_rate=learning_rate,
            optim=optim,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            use_cpu=not have_cuda,
            bf16_full_eval=have_cuda,
            bf16=have_cuda,
        )

    return args


def get_trainer(
    model: Module,
    data: DatasetDict,
    arch: Architecture,
    collate_fn: DataCollator,
    metrics_fn: Callable[[EvalPrediction, bool], dict[str, int | float]],
    *,
    do_eval: bool,
    early_stopping: bool,
    epochs: int = 0,
    learning_rate: float = 5e-5,
    batch_size: int = 8,
    run_name: str = "default",
    report_to: str = "none",
) -> Trainer:
    """Build a Trainer for the architecture, adding early stopping if requested."""
    args = get_args(
        arch,
        do_eval,
        epochs=epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        run_name=run_name,
        report_to=report_to,
    )

    _metrics_fn = cast("Callable", metrics_fn)
    train_dataset = data.get("train")
    eval_dataset = cast("Dataset", data.get("validation"))

    config = cast("Module", model.config)
    if isinstance(model, PeftModel) and config.model_type == "gemma3":
        logger.debug("patch pt-gemma3 forward pass")
        _patch_gemma3(model)

    if config.model_type == "gemma3":
        logger.debug("use gemma3 trainer")
        trainer_cls = Gemma3Trainer
    elif arch == "encoder-decoder":
        logger.debug("use seq2seq trainer")
        trainer_cls = Seq2SeqTrainer
        collate_fn = StripTokenTypeIds(collate_fn)
    else:
        logger.debug("use regular trainer")
        trainer_cls = Trainer

    trainer = trainer_cls(
        args=args,
        model=model,
        data_collator=collate_fn,
        eval_dataset=eval_dataset,
        train_dataset=train_dataset,
        compute_metrics=_metrics_fn,
    )

    if not do_eval and early_stopping:
        logger.warning("not using early stopping because not running evaluation")
    elif do_eval and early_stopping:
        patience = 4
        tolerance = 0.01
        logger.info(
            "using early stopping with %d patience and %.2f tolerance",
            patience,
            tolerance,
        )

        trainer.add_callback(EarlyStoppingCallback(patience, tolerance))

    return trainer


def save_model(
    model: PreTrainedModel | PeftModel, trainer: Trainer, run_name: str
) -> None:
    """Save the trained model, falling back to `run_name` so it is never lost."""
    output_dir = trainer.args.output_dir
    if output_dir is None:
        output_dir = str(LOGDIR / run_name)
        logger.warning(
            "no trainer arguments output_dir configured, saving to '%s'",
            output_dir,
        )

    logger.info("save model to '%s'", output_dir)
    model.save_pretrained(output_dir)
