"""Fine-tune a pretrained model and run test evaluation."""

from typing import TYPE_CHECKING, cast

import mlflow
from transformers import AutoConfig

from saspbft.datasets.registry import load_data
from saspbft.logging import logger
from saspbft.modeling.arch import get_arch
from saspbft.modeling.collate import get_collator
from saspbft.modeling.loading import get_model
from saspbft.modeling.metrics import get_metrics_fn
from saspbft.modeling.tokenizer import load_tokenizer
from saspbft.modeling.trainer import (
    RequeueOnSignal,
    find_checkpoint,
    get_trainer,
    train_or_requeue,
)
from saspbft.scripts.tracking import run_name, start_run

if TYPE_CHECKING:
    from torch.utils.data import Dataset

    from saspbft.types import Architecture, DatasetName


def fine_tune(
    model_path: str,
    dataset: DatasetName,
    head_only: bool,
    n_shot: int,
    *,
    arch: Architecture | None = None,
    train_samples: int | None,
    val_samples: int | None,
    do_eval: bool,
    early_stopping: bool,
    resume: bool = True,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    mlflow_experiment: str,
    mlflow_run_name: str | None,
    mlflow_tracking_uri: str,
) -> None:
    """Fine-tune `model_path` on `dataset` and evaluate on the test split."""
    logger.info("load config for '%s'", model_path)
    config = AutoConfig.from_pretrained(model_path)
    arch = get_arch(config, arch)

    tokenizer = load_tokenizer(model_path)
    collate_fn = get_collator(tokenizer, arch)
    metrics_fn = get_metrics_fn(tokenizer, arch)
    data, info = load_data(
        tokenizer,
        dataset,
        arch,
        n_shot,
        n_train_samples=train_samples,
        n_val_samples=val_samples,
    )

    if dataset in {"boolq", "wic"}:
        logger.warning("using superglue validation data for test, labels are private")
        data["test"] = data["validation"]

    logger.info("load '%s'", model_path)
    model = get_model(tokenizer, model_path, info, arch, head_only)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    method = "cls-head" if head_only else "fine-tune"
    if mlflow_run_name is None:
        mlflow_run_name = run_name(dataset, model_path, method, train_samples)

    logger.info("total parameters %d", total)
    logger.info("trainable parameters %d", trainable)
    logger.info("tracking '%s' of experiment '%s'", mlflow_run_name, mlflow_experiment)

    checkpoint = find_checkpoint(mlflow_run_name) if resume else None

    start_run(
        mlflow_experiment,
        mlflow_run_name,
        mlflow_tracking_uri,
        resume=checkpoint is not None,
    )

    mlflow.log_param("n_shot", n_shot)
    mlflow.log_param("dataset", dataset)
    mlflow.log_param("architecture", arch)
    mlflow.log_param("head_only", head_only)
    mlflow.log_param("base_model", model_path)
    mlflow.log_param("system_prompt", info["system_prompt"])
    mlflow.log_param("method", method)
    mlflow.log_metric("train_samples", len(data["train"]))
    mlflow.log_metric("validation_samples", len(data["validation"]) if do_eval else 0)
    mlflow.log_metric("test_samples", len(data["test"]))
    mlflow.log_metric("total_parameters", total)
    mlflow.log_metric("trainable_parameters", trainable)

    on_signal = RequeueOnSignal()
    trainer = get_trainer(
        model=model,
        data=data,
        arch=arch,
        collate_fn=collate_fn,
        metrics_fn=metrics_fn,
        do_eval=do_eval,
        early_stopping=early_stopping,
        epochs=epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        run_name=mlflow_run_name,
        report_to="mlflow",
        checkpoints=True,
        callbacks=(on_signal,),
    )

    if not train_or_requeue(trainer, on_signal, checkpoint):
        return

    logger.debug("start test eval")
    test = cast("Dataset", data["test"])
    trainer.evaluate(test, metric_key_prefix="test")

    mlflow.end_run()
