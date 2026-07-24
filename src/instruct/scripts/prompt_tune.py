"""Prompt-tune a pretrained model and run test evaluation."""

from typing import TYPE_CHECKING, cast

import mlflow
from transformers import AutoConfig

from instruct.datasets.util import get_collator, load_data, load_tokenizer
from instruct.logging import logger
from instruct.metrics import get_metrics_fn
from instruct.modeling import get_arch, get_pt_model, get_trainer

if TYPE_CHECKING:
    from peft import PromptTuningConfig
    from torch.utils.data import Dataset
    from transformers.trainer import Trainer

    from instruct.types import Architecture, DatasetName, PrefixInit


def _output_dir(trainer: Trainer) -> str:
    """Return the trainer's configured output dir, raising if unset."""
    logdir = trainer.args.output_dir
    if logdir is None:
        msg = "no trainer arguments logdir configured"
        raise RuntimeError(msg)
    return logdir


def prompt_tune(
    model_path: str,
    dataset: DatasetName,
    prefix_init: PrefixInit,
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
    """Prompt-tune `model_path` on `dataset` and evaluate on the test split."""
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

    logger.info(
        "get prompt tuning model for '%s' with %s prefix initialization",
        model_path,
        prefix_init,
    )

    model = get_pt_model(prefix_init, tokenizer, model_path, arch, info)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    ptcfg = cast("PromptTuningConfig", model.peft_config["default"])

    if run_name is None:
        samples = n_train_samples or "all"
        run_name = f"{dataset}/{samples}/{model_path}/{prefix_init}-prefix"

    logger.info("total parameters %d", total)
    logger.info("trainable parameters %d", trainable)
    logger.info("virtual tokens %d", ptcfg.num_virtual_tokens)
    logger.info("tracking '%s' of experiment '%s'", run_name, experiment)

    mlflow.set_experiment(experiment)
    mlflow.start_run(run_name=run_name)
    mlflow.log_param("n_shot", n_shot)
    mlflow.log_param("dataset", dataset)
    mlflow.log_param("architecture", arch)
    mlflow.log_param("base_model", model_path)
    mlflow.log_param("prefix_init", prefix_init)
    mlflow.log_param("system_prompt", info["system_prompt"])
    mlflow.log_param("method", f"prompt-tune-{prefix_init}")
    mlflow.log_param("num_virtual_tokens", ptcfg.num_virtual_tokens)
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

    logger.info("save peft adapter to %s", trainer.args.output_dir)
    model.save_pretrained(_output_dir(trainer))

    mlflow.end_run()
