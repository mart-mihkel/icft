"""Trainer construction, training arguments, and Gemma 3 / seq2seq quirks."""

import signal
from typing import TYPE_CHECKING, Any, cast

import torch
from peft import PeftModel
from transformers import (
    DataCollator,
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
)
from transformers.trainer import Trainer
from transformers.trainer_utils import get_last_checkpoint
from transformers.training_args import TrainingArguments

from saspbft.constants import LOGDIR
from saspbft.logging import logger
from saspbft.slurm import requeue

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import FrameType

    from datasets.dataset_dict import DatasetDict
    from torch import Tensor
    from torch.nn import Module
    from torch.utils.data import Dataset
    from transformers import EvalPrediction, TrainerControl, TrainerState
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


class RequeueOnSignal(TrainerCallback):
    """
    Checkpoint and stop training when SLURM signals the time limit is near.

    SLURM sends the signal configured by `sbatch --signal` before killing the
    job. Stopping at the next step boundary leaves a fresh checkpoint behind,
    so the requeued job resumes from where it left off instead of the last
    epoch.
    """

    def __init__(self, signum: int = signal.SIGUSR1) -> None:
        """Store the signal to listen for."""
        self.signum = signum
        self.signalled = False

    def _handle(self, signum: int, frame: FrameType | None) -> None:
        del frame
        logger.warning("caught signal %d, stopping at the next step", signum)
        self.signalled = True

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: object,
    ) -> None:
        """Install the handler once training owns the process."""
        del args, state, control, kwargs
        signal.signal(self.signum, self._handle)

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: object,
    ) -> None:
        """Ask the trainer to save and stop if the signal arrived."""
        del args, state, kwargs
        if not self.signalled:
            return

        control.should_save = True
        control.should_training_stop = True


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
    checkpoints: bool = False,
) -> TrainingArguments:
    """Build Trainer arguments, using seq2seq args for encoder-decoder models."""
    have_cuda = torch.cuda.is_available()
    optim = "adamw_8bit" if have_cuda else "adamw_torch_fused"
    eval_strategy = "epoch" if do_eval else "no"
    save_strategy = "epoch" if checkpoints else "no"
    out_dir = LOGDIR / run_name

    if arch == "encoder-decoder":
        logger.debug("use seq2seq training args")
        args = Seq2SeqTrainingArguments(
            run_name=run_name,
            report_to=report_to,
            output_dir=str(out_dir),
            save_strategy=save_strategy,
            save_total_limit=1,
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
            save_strategy=save_strategy,
            save_total_limit=1,
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
    checkpoints: bool = False,
    callbacks: tuple[TrainerCallback, ...] = (),
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
        checkpoints=checkpoints,
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
        callbacks=list(callbacks),
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


def train_or_requeue(
    trainer: Trainer,
    on_signal: RequeueOnSignal,
    checkpoint: str | None,
) -> bool:
    """
    Train from `checkpoint`, reporting whether training ran to completion.

    A signalled run leaves its tracking run open and requeues instead, so the
    next job reattaches to it and resumes from the checkpoint just written.
    """
    logger.debug("start trainer")
    trainer.train(resume_from_checkpoint=checkpoint)

    if not on_signal.signalled:
        return True

    logger.warning("stopped early, leaving the run open for the requeued job")
    requeue()
    return False


def find_checkpoint(run_name: str) -> str | None:
    """Find the latest checkpoint of a previous run of `run_name`, if any exists."""
    out_dir = LOGDIR / run_name
    if not out_dir.is_dir():
        logger.warning("no output dir '%s', training from scratch", out_dir)
        return None

    checkpoint = get_last_checkpoint(str(out_dir))
    if checkpoint is None:
        logger.warning("no checkpoint in '%s', training from scratch", out_dir)
        return None

    logger.info("resuming from checkpoint '%s'", checkpoint)
    return checkpoint
