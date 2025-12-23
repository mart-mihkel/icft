import logging

from datasets.arrow_dataset import Dataset
from datasets.load import load_dataset
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments

from icftner.datasets.squad import squad_collate_fn, tokenize_squad
from icftner.models.bert import freeze_bert

logger = logging.getLogger(__name__)


def main(
    pretrained_model: str,
    out_dir: str,
    epochs: int,
    head_only: bool,
    train_split: str,
    eval_split: str,
):
    logger.info("load squad")
    train, eval = load_dataset("squad", split=[train_split, eval_split])
    assert isinstance(train, Dataset)
    assert isinstance(eval, Dataset)

    logger.info("load tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)

    logger.info("tokenize squad")
    train_tokenized = tokenize_squad(tokenizer, train)
    eval_tokenized = tokenize_squad(tokenizer, eval)

    logger.info("load %s", pretrained_model)
    bert = AutoModelForQuestionAnswering.from_pretrained(pretrained_model)

    logger.info("init trainer")
    args = TrainingArguments(
        output_dir=out_dir,
        overwrite_output_dir=True,
        logging_dir=f"{out_dir}/tensorboard",
        logging_steps=2500,
        report_to="tensorboard",
        num_train_epochs=epochs,
        eval_strategy="steps",
        eval_steps=5000,
        save_strategy="epoch",
        auto_find_batch_size=True,
        fp16=True,
    )

    if head_only:
        head_params = ["qa_outputs.bias", "qa_outputs.weight"]
        freeze_bert(bert, skip_params=head_params)

    trainer = Trainer(
        bert,
        args=args,
        train_dataset=train_tokenized,
        eval_dataset=eval_tokenized,
        data_collator=squad_collate_fn,
    )

    trainer.train()
