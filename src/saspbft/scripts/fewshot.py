"""Run few-shot test evaluation for a pretrained model."""

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
from saspbft.modeling.trainer import get_trainer
from saspbft.scripts.tracking import run_name

if TYPE_CHECKING:
    from datasets.splits import Split
    from torch.utils.data import Dataset

    from saspbft.types import Architecture, DatasetName


def few_shot(
    model_path: str,
    dataset: DatasetName,
    n_shot: int,
    *,
    arch: Architecture | None = None,
    batch_size: int,
    mlflow_experiment: str,
    mlflow_run_name: str | None,
    mlflow_tracking_uri: str,
) -> None:
    """Run few-shot test evaluation of `model_path` on `dataset`."""
    logger.info("load config for '%s'", model_path)
    config = AutoConfig.from_pretrained(model_path)
    arch = get_arch(config, arch)

    tokenizer = load_tokenizer(model_path)
    collate_fn = get_collator(tokenizer, arch)
    metrics_fn = get_metrics_fn(tokenizer, arch)

    split = cast("Split", {"test": "test"})
    data, info = load_data(tokenizer, dataset, arch, n_shot, split=split)

    if dataset in {"boolq", "wic"}:
        logger.warning("using superglue validation data, test labels are private")
        data["test"] = data["validation"]

    logger.info("load '%s'", model_path)
    model = get_model(tokenizer, model_path, info, arch, head_only=False)

    total = sum(p.numel() for p in model.parameters())

    method = f"{n_shot}-shot"
    if mlflow_run_name is None:
        mlflow_run_name = run_name(dataset, model_path, method, train_samples=0)

    logger.info("tracking '%s' of experiment '%s'", mlflow_run_name, mlflow_experiment)

    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(mlflow_experiment)
    mlflow.start_run(run_name=mlflow_run_name)
    mlflow.log_param("n_shot", n_shot)
    mlflow.log_param("dataset", dataset)
    mlflow.log_param("architecture", arch)
    mlflow.log_param("base_model", model_path)
    mlflow.log_param("method", method)
    mlflow.log_param("system_prompt", info["system_prompt"])
    mlflow.log_metric("train_samples", 0)
    mlflow.log_metric("validation_samples", 0)
    mlflow.log_metric("test_samples", len(data["test"]))
    mlflow.log_metric("total_parameters", total)
    mlflow.log_metric("trainable_parameters", 0)

    trainer = get_trainer(
        model=model,
        data=data,
        arch=arch,
        collate_fn=collate_fn,
        metrics_fn=metrics_fn,
        do_eval=False,
        early_stopping=False,
        batch_size=batch_size,
        run_name=mlflow_run_name,
        report_to="mlflow",
    )

    logger.debug("start test eval")
    test = cast("Dataset", data["test"])
    trainer.evaluate(test, metric_key_prefix="test")

    mlflow.end_run()
