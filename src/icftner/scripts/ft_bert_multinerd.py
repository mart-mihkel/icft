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
    MULTINERD_TAG2ID,
    compute_multinerd_prompted_metrics,
    filter_multinerd_english,
    tokenize_multinerd_prompted,
)
from icftner.models.bert import freeze_bert

logger = logging.getLogger(__name__)


def main(
    pretrained_model: str,
    out_dir: str,
    epochs: int,
    head_only: bool,
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

    if head_only:
        head_params = [
            "classifier.bias",
            "classifier.weight",
            "pre_classifier.bias",
            "pre_classifier.weight",
        ]

        freeze_bert(bert, skip_params=head_params)

    logger.info("init trainer")
    args = TrainingArguments(
        output_dir=out_dir,
        overwrite_output_dir=True,
        logging_dir=f"{out_dir}/tensorboard",
        logging_steps=5000,
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
        bert,
        args=args,
        train_dataset=train_tokenized,
        eval_dataset=eval_tokenized,
        data_collator=data_collator,
        compute_metrics=compute_multinerd_prompted_metrics,
    )

    trainer.train()
