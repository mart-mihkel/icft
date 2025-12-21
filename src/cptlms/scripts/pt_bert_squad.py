import logging

from datasets.arrow_dataset import Dataset
from datasets.load import load_dataset
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments

from cptlms.datasets.squad import squad_collate_fn, tokenize_squad
from cptlms.models.bert import PTuningBertQuestionAnswering
from cptlms.models.prompt_encoder import EncoderReparameterizationType

logger = logging.getLogger(__name__)


def main(
    pretrained_model: str,
    out_dir: str,
    epochs: int,
    num_virtual_tokens: int,
    train_new_layers: bool,
    encoder_hidden_size: int,
    encoder_reparam_type: EncoderReparameterizationType,
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

    logger.info("init ptunging %s", pretrained_model)
    pt_bert = PTuningBertQuestionAnswering(
        bert=bert,
        num_virtual_tokens=num_virtual_tokens,
        train_new_layers=train_new_layers,
        encoder_hidden_size=encoder_hidden_size,
        encoder_reparam_type=encoder_reparam_type,
    )

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

    trainer = Trainer(
        pt_bert,
        args=args,
        train_dataset=train_tokenized,
        eval_dataset=eval_tokenized,
        data_collator=squad_collate_fn,
    )

    trainer.train()
