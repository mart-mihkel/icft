from typing import cast

import mlflow
from datasets.splits import Split
from torch.utils.data import Dataset
from transformers import AutoConfig

from instruct.datasets.util import get_collator, load_data, load_tokenizer
from instruct.logging import logger
from instruct.metrics import get_metrics_fn
from instruct.modeling import get_arch, get_model, get_trainer
from instruct.types import DatasetName


def few_shot(
    model_path: str,
    dataset: DatasetName,
    n_shot: int,
    batch_size: int,
    experiment: str,
    run_name: str | None,
) -> None:
    logger.info("load config for '%s'", model_path)
    config = AutoConfig.from_pretrained(model_path)
    arch = get_arch(config)

    tokenizer = load_tokenizer(model_path)
    collate_fn = get_collator(tokenizer, arch)
    metrics_fn = get_metrics_fn(tokenizer, arch)

    split = cast(Split, {"test": "test"})
    data, info = load_data(tokenizer, dataset, arch, n_shot, split=split)

    if dataset == "boolq" or dataset == "wic":
        logger.warning("using superglue dev data, test labels are private")
        data["test"] = data["dev"]

    logger.info("load '%s'", model_path)
    model = get_model(tokenizer, model_path, info, arch, head_only=False)

    total = sum(p.numel() for p in model.parameters())

    if run_name is None:
        run_name = f"{dataset}/all/{model_path}/few-shot"

    logger.info("tracking '%s' of experiment '%s'", run_name, experiment)

    mlflow.set_experiment(experiment)
    mlflow.start_run(run_name=run_name)
    mlflow.log_param("n_shot", n_shot)
    mlflow.log_param("dataset", dataset)
    mlflow.log_param("architecture", arch)
    mlflow.log_param("base_model", model_path)
    mlflow.log_param("method", f"{n_shot}-shot")
    mlflow.log_param("system_prompt", info["system_prompt"])
    mlflow.log_metric("train_samples", 0)
    mlflow.log_metric("dev_samples", 0)
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
        batch_size=batch_size,
        run_name=run_name,
        report_to="mlflow",
    )

    logger.debug("start test eval")
    test = cast(Dataset, data["test"])
    trainer.evaluate(test, metric_key_prefix="test")

    mlflow.end_run()
