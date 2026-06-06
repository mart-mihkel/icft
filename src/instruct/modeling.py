from collections.abc import Callable
from typing import Any, cast

import torch
from datasets.dataset_dict import DatasetDict
from peft import PeftModel, PromptTuningConfig, TaskType, get_peft_model
from torch import Tensor
from torch.nn import Module
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    DataCollator,
    DistilBertModel,
    EarlyStoppingCallback,
    EvalPrediction,
    PreTrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizerFast,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    T5Gemma2EncoderConfig,
    T5Gemma2Model,
    T5Gemma2TextConfig,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
from transformers.models.gemma3.modeling_gemma3 import Gemma3ModelOutputWithPast
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments

from instruct.constants import (
    decoder_model_types,
    encoder_decoder_model_types,
    encoder_model_types,
    logdir,
)
from instruct.datasets.util import DatasetInfo
from instruct.logging import logger
from instruct.types import Architecture, PrefixInit


class LoggerCallback(TrainerCallback):
    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: dict,
    ) -> None:
        args, state, control

        logs = kwargs.get("logs")
        if logs:
            logger.info(logs)


class Gemma3Trainer(Trainer):
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
    def __init__(self, collator: DataCollator) -> None:
        self.collator = collator

    def __call__(self, features: list[dict]) -> dict:
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

    def _gemma3_patched_forward(*args, **kwargs) -> tuple | Gemma3ModelOutputWithPast:
        ref = kwargs["attention_mask"]
        kwargs["token_type_ids"] = torch.zeros_like(ref)
        return _original_forward(*args, **kwargs)

    _base_model.forward = _gemma3_patched_forward


def get_arch(config: PreTrainedConfig) -> Architecture:
    mt = config.model_type
    if config.is_encoder_decoder or mt in encoder_decoder_model_types:
        arch = "encoder-decoder"
    elif getattr(config, "is_decoder", False) or mt in decoder_model_types:
        arch = "decoder"
    elif mt in encoder_model_types:
        arch = "encoder"
    else:
        raise RuntimeError(f"failed to infer architecture for '{mt}'")

    logger.debug("inferred model architecture '%s' for '%s'", arch, mt)
    return arch


def get_model(
    tokenizer: PreTrainedTokenizerFast,
    model_path: str,
    data_info: DatasetInfo,
    arch: Architecture,
    head_only: bool,
) -> PreTrainedModel:
    if torch.cuda.is_available():
        dtype = torch.bfloat16
        logger.debug("using half precision brain float")
    else:
        dtype = torch.float32
        logger.debug("using full precision floating point")

    skip_freeze = None
    if arch == "encoder":
        logger.debug("load '%s' for sequence classification", model_path)
        model, loading_info = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            output_loading_info=True,
            num_labels=len(data_info["id2label"]),
            id2label=data_info["id2label"],
            label2id=data_info["label2id"],
            device_map="auto",
            dtype=dtype,
        )

        skip_freeze = loading_info["missing_keys"]
    elif arch == "decoder":
        logger.debug("load '%s' for causal language modeling", model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            dtype=dtype,
        )
    elif arch == "encoder-decoder":
        logger.debug("load '%s' for sequence to sequence", model_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_path,
            device_map="auto",
            dtype=dtype,
        )

    config = model.config
    if "text_config" in config:
        logger.debug("using text config of multimodal model")
        config = config.text_config

    if config.pad_token_id is None:
        logger.warning("model doesn't have a padding token, using eos")
        config.pad_token_id = tokenizer.eos_token_id

    if head_only:
        freeze(model, skip_freeze)

    return model


def _distiblert_prompt_tuning_kwargs(model: DistilBertModel) -> dict[str, Any]:
    logger.debug("get prompt tuning args for distilbert")

    cfg = model.config
    return {
        "token_dim": cfg.dim,
        "num_layers": cfg.n_layers,
        "num_attention_heads": cfg.n_heads,
    }


def _t5gemma2_prompt_tuning_kwargs(model: T5Gemma2Model) -> dict[str, Any]:
    logger.debug("get prompt tuning args for t5gemma-2")

    cfg = cast(T5Gemma2EncoderConfig, model.config.encoder)
    cfg = cast(T5Gemma2TextConfig, cfg.text_config)
    return {
        "token_dim": cfg.hidden_size,
        "num_layers": cfg.num_hidden_layers,
        "num_attention_heads": cfg.num_attention_heads,
    }


def get_pt_model(
    prefix_init: PrefixInit,
    tokenizer: PreTrainedTokenizerFast,
    model_path: str,
    arch: Architecture,
    data_info: DatasetInfo,
) -> PeftModel:
    sys_prompt = data_info["system_prompt"]
    if tokenizer.chat_template is None:
        sys_enc = tokenizer(sys_prompt, truncation=True)
    else:
        conv = [{"role": "system", "content": sys_prompt}]
        sys_enc = tokenizer.apply_chat_template(conv, truncation=True)

    num_virtual_tokens = len(cast(dict[str, list[int]], sys_enc)["input_ids"])
    base = get_model(tokenizer, model_path, data_info, arch, head_only=False)

    if arch == "encoder":
        task_type = TaskType.SEQ_CLS
    elif arch == "decoder":
        task_type = TaskType.CAUSAL_LM
    elif arch == "encoder-decoder":
        task_type = TaskType.SEQ_2_SEQ_LM

    if prefix_init == "pretrained":
        init = "TEXT"
    elif prefix_init == "random":
        init = "RANDOM"

    special_kwargs = {}
    if "distilbert" in model_path:
        base = cast(DistilBertModel, base)
        special_kwargs = _distiblert_prompt_tuning_kwargs(base)
    elif "t5gemma-2" in model_path:
        base = cast(T5Gemma2Model, base)
        special_kwargs = _t5gemma2_prompt_tuning_kwargs(base)

    config = PromptTuningConfig(
        task_type=task_type,
        prompt_tuning_init=init,
        tokenizer_name_or_path=model_path,
        prompt_tuning_init_text=sys_prompt,
        num_virtual_tokens=num_virtual_tokens,
        **special_kwargs,
    )

    logger.debug("get peft model for '%s'", model_path)
    return cast(PeftModel, get_peft_model(base, config))


def freeze(model: Module, skip: set[str] | None = None) -> None:
    if skip is None:
        skip = set()

    logger.info("freeze base model")
    for name, param in model.named_parameters():
        if name in skip:
            logger.info("skip '%s'", name)
            continue

        param.requires_grad = False


def get_args(
    arch: Architecture,
    do_eval: bool,
    epochs: int = 0,
    learning_rate: float = 5e-5,
    batch_size: int = 8,
    run_name: str = "default",
    report_to: str = "none",
) -> TrainingArguments:
    have_cuda = torch.cuda.is_available()
    optim = "adamw_8bit" if have_cuda else "adamw_torch_fused"
    eval_strategy = "epoch" if do_eval else "no"
    out_dir = logdir / run_name

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
    do_eval: bool,
    early_stopping: bool,
    epochs: int = 0,
    learning_rate: float = 5e-5,
    batch_size: int = 8,
    run_name: str = "default",
    report_to: str = "none",
) -> Trainer:
    args = get_args(
        arch,
        do_eval,
        epochs,
        learning_rate,
        batch_size,
        run_name,
        report_to,
    )

    _metrics_fn = cast(Callable, metrics_fn)
    train_dataset = data.get("train")
    eval_dataset = cast(Dataset, data.get("dev"))

    config = cast(Module, model.config)
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
