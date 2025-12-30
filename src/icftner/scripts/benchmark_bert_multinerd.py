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

logger = logging.getLogger(__name__)


def main(
    pretrained_model: str,
    out_dir: str,
    english_only: bool,
    test_split: str,
):
    logger.info("load multinerd")
    eval = load_dataset(
        "Babelscape/multinerd",
        split=test_split,
        verification_mode=VerificationMode.NO_CHECKS,
    )

    assert isinstance(eval, Dataset)

    logger.info("load tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)

    if english_only:
        logger.info("filter multinerd english")
        eval = eval.filter(filter_multinerd_english, batched=True)

    logger.info("tokenize multinerd prompted")
    system_tokens = (
        "Task : Determine the named entity tag . Question: What is the NER tag"
        " of the given worn in the following sentence ? Possible tags :".split()
        + list(MULTINERD_ID2TAG.values())
    )

    eval_tokenized = tokenize_multinerd_prompted(
        tokenizer=tokenizer,
        data=eval,
        system_prompt_tokens=system_tokens,  # type: ignore
    )

    logger.info("load %s", pretrained_model)
    bert = AutoModelForSequenceClassification.from_pretrained(
        pretrained_model,
        num_labels=len(MULTINERD_ID2TAG),
        id2label=MULTINERD_ID2TAG,
        label2id=MULTINERD_TAG2ID,
    )

    logger.info("init trainer")
    args = TrainingArguments(
        output_dir=out_dir,
        overwrite_output_dir=True,
        logging_dir=f"{out_dir}/tensorboard",
        report_to="tensorboard",
        auto_find_batch_size=True,
        fp16=True,
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    trainer = Trainer(
        bert,
        args=args,
        eval_dataset=eval_tokenized,
        data_collator=data_collator,
        compute_metrics=compute_multinerd_prompted_metrics,
    )

    trainer.evaluate()
