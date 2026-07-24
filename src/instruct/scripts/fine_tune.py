"""Fine-tune a pretrained model and run test evaluation."""

from typing import TYPE_CHECKING, cast

import mlflow
from transformers import AutoConfig

from instruct.datasets.util import get_collator, load_data, load_tokenizer
from instruct.logging import logger
from instruct.metrics import get_metrics_fn
from instruct.modeling import get_arch, get_model, get_trainer

if TYPE_CHECKING:
    from torch.utils.data import Dataset

    from instruct.types import Architecture, DatasetName


def fine_tune(
    model_path: str,
    dataset: DatasetName,
    head_only: bool,
    n_shot: int,
    *,
    arch: Architecture | None = None,
    n_train_samples: int | None,
    n_dev_samples: int | None,
    do_eval: bool,
    early_stopping: bool,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    experiment: str,
    run_name: str | None,
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
        n_train_samples=n_train_samples,
        n_dev_samples=n_dev_samples,
    )

    if dataset in {"boolq", "wic"}:
        logger.warning("using superglue dev data for test, labels are private")
        data["test"] = data["dev"]

    logger.info("load '%s'", model_path)
    model = get_model(tokenizer, model_path, info, arch, head_only)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if run_name is None:
        ft_task = "cls-head" if head_only else "fine-tune"
        samples = n_train_samples or "all"
        run_name = f"{dataset}/{samples}/{model_path}/{ft_task}"

    logger.info("total parameters %d", total)
    logger.info("trainable parameters %d", trainable)
    logger.info("tracking '%s' of experiment '%s'", run_name, experiment)

    mlflow.set_experiment(experiment)
    mlflow.start_run(run_name=run_name)
    mlflow.log_param("n_shot", n_shot)
    mlflow.log_param("dataset", dataset)
    mlflow.log_param("architecture", arch)
    mlflow.log_param("head_only", head_only)
    mlflow.log_param("base_model", model_path)
    mlflow.log_param("system_prompt", info["system_prompt"])
    mlflow.log_param("method", "cls-head" if head_only else "fine-tune")
    mlflow.log_metric("train_samples", len(data["train"]))
    mlflow.log_metric("dev_samples", len(data["dev"]) if do_eval else 0)
    mlflow.log_metric("test_samples", len(data["test"]))
    mlflow.log_metric("total_parameters", total)
    mlflow.log_metric("trainable_parameters", trainable)

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
        run_name=run_name,
        report_to="mlflow",
    )

    logger.debug("start trainer")
    trainer.train()

    logger.debug("start test eval")
    test = cast("Dataset", data["test"])
    trainer.evaluate(test, metric_key_prefix="test")

    mlflow.end_run()
