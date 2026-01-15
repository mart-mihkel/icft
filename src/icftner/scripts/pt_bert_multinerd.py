import logging

from datasets.arrow_dataset import Dataset
from datasets.load import load_dataset
from datasets.utils.info_utils import VerificationMode
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
)
from transformers.models.auto.modeling_auto import (
    AutoModelForSequenceClassification,
)
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments

from icftner.datasets.multinerd import (
    MULTINERD_ID2TAG,
    MULTINERD_SYSTEM_PROMPT,
    MULTINERD_TAG2ID,
    compute_multinerd_prompted_metrics,
    filter_multinerd_english,
    tokenize_multinerd_prompted,
)
from icftner.models.bert import PTuningBertSequenceClassification
from icftner.models.prompt_encoder import EncoderReparameterizationType

logger = logging.getLogger(__name__)


def main(
    pretrained_model: str,
    out_dir: str,
    epochs: int,
    num_virtual_tokens: int,
    auto_virtual_tokens: bool,
    train_new_layers: bool,
    encoder_hidden_size: int,
    encoder_random_init: bool,
    encoder_reparam_type: EncoderReparameterizationType,
    english_only: bool,
    train_split: str,
    eval_split: str,
):
    logger.info("load multinerd")
    train, eval = load_dataset(
        "Babelscape/multinerd",
        split=[train_split, eval_split],
        verification_mode=VerificationMode.NO_CHECKS,
    )

    assert isinstance(train, Dataset)
    assert isinstance(eval, Dataset)

    logger.info("load tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)

    if english_only:
        logger.info("filter multinerd english")
        train = train.filter(filter_multinerd_english, batched=True)
        eval = eval.filter(filter_multinerd_english, batched=True)

    logger.info("tokenize multinerd prompted")
    train_tokenized = tokenize_multinerd_prompted(tokenizer=tokenizer, data=train)
    eval_tokenized = tokenize_multinerd_prompted(tokenizer=tokenizer, data=eval)

    logger.info("load %s", pretrained_model)
    bert = AutoModelForSequenceClassification.from_pretrained(
        pretrained_model,
        num_labels=len(MULTINERD_ID2TAG),
        id2label=MULTINERD_ID2TAG,
        label2id=MULTINERD_TAG2ID,
    )

    logger.info("init ptunging %s", pretrained_model)
    system_prompt = tokenizer(MULTINERD_SYSTEM_PROMPT, is_split_into_words=True)
    system_prompt_ids = system_prompt["input_ids"]

    if auto_virtual_tokens:
        num_virtual_tokens = len(system_prompt_ids)

    encoder_prompt_token_ids = None
    if not encoder_random_init:
        encoder_prompt_token_ids = system_prompt_ids

    pt_bert = PTuningBertSequenceClassification(
        bert=bert,
        num_virtual_tokens=num_virtual_tokens,
        train_new_layers=train_new_layers,
        encoder_hidden_size=encoder_hidden_size,
        encoder_random_init=encoder_random_init,
        encoder_reparam_type=encoder_reparam_type,
        encoder_prompt_token_ids=encoder_prompt_token_ids,
    )

    total_params = sum(p.numel() for p in pt_bert.parameters())
    trainable_params = sum(p.numel() for p in pt_bert.parameters() if p.requires_grad)
    logger.info("total params:     %d", total_params)
    logger.info("trainable params: %d", trainable_params)

    logger.info("init trainer")
    args = TrainingArguments(
        output_dir=out_dir,
        overwrite_output_dir=True,
        logging_dir=f"{out_dir}/tensorboard",
        logging_steps=5000,
        logging_first_step=True,
        report_to="tensorboard",
        num_train_epochs=epochs,
        eval_strategy="steps",
        eval_steps=10000,
        save_strategy="epoch",
        auto_find_batch_size=True,
        fp16=True,
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    trainer = Trainer(
        pt_bert,
        args=args,
        train_dataset=train_tokenized,
        eval_dataset=eval_tokenized,
        data_collator=data_collator,
        compute_metrics=compute_multinerd_prompted_metrics,
    )

    trainer.train()
