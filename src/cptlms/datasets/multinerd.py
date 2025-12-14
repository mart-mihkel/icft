import logging
from typing import Literal, TypedDict

from datasets.arrow_dataset import Dataset
from datasets.load import load_dataset
from datasets.utils.info_utils import VerificationMode
from transformers import BatchEncoding
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

logger = logging.getLogger("cptlms")


type MultinerdLang = Literal[
    "zh",
    "nl",
    "en",
    "fr",
    "de",
    "it",
    "pl",
    "pt",
    "ru",
    "es",
]

type MultinerdTag = Literal[
    "O",
    "B-PER",
    "I-PER",
    "B-ORG",
    "I-ORG",
    "B-LOC",
    "I-LOC",
    "B-ANIM",
    "I-ANIM",
    "B-BIO",
    "I-BIO",
    "B-CEL",
    "I-CEL",
    "B-DIS",
    "I-DIS",
    "B-EVE",
    "I-EVE",
    "B-FOOD",
    "I-FOOD",
    "B-INST",
    "I-INST",
    "B-MEDIA",
    "I-MEDIA",
    "B-MYTH",
    "I-MYTH",
    "B-PLANT",
    "I-PLANT",
    "B-TIME",
    "I-TIME",
    "B-VEHI",
    "I-VEHI",
]


class MultinerdBatch(TypedDict):
    tokens: list[list[str]]
    ner_tags: list[list[MultinerdTag]]
    lang: list[MultinerdLang]


class Multinerd:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerFast,
        train_split: str = "train",
        val_split: str = "validation",
        test_split: str = "test",
        english_only: bool = True,
    ) -> None:
        logger.info("load multinerd")
        train, val, test = load_dataset(
            "Babelscape/multinerd",
            split=[train_split, val_split, test_split],
            verification_mode=VerificationMode.NO_CHECKS,
        )

        assert isinstance(train, Dataset)
        assert isinstance(val, Dataset)
        assert isinstance(test, Dataset)

        self.train = train
        self.val = val
        self.test = test

        if english_only:
            logger.info("filter multinerd english")
            self.train = train.filter(_filter_en, batched=True)
            self.val = val.filter(_filter_en, batched=True)
            self.test = test.filter(_filter_en, batched=True)

        logger.info("tokenize multinerd")
        self.tokenizer = tokenizer
        self.train_tokenized = self.train.map(self._tokenize, batched=True)
        self.val_tokenized = self.val.map(self._tokenize, batched=True)
        self.test_tokenized = self.test.map(self._tokenize, batched=True)

    def _tokenize(self, batch: MultinerdBatch) -> BatchEncoding:
        """
        Args:
            batch (MultinerdBatch)

        Returns:
            BatchEncoding:
                input_ids: list[list[int]]
                attention_map: list[list[int]]
                labels: list[list[int]]
        """
        tokenized = self.tokenizer(
            batch["tokens"],
            truncation=True,
            is_split_into_words=True,
        )

        labels = []
        for i, ner_tags in enumerate(batch["ner_tags"]):
            word_ids = tokenized.word_ids(batch_index=i)
            tag_ids = _align_words_to_tags(word_ids=word_ids, ner_tags=ner_tags)
            labels.append(tag_ids)

        tokenized["labels"] = labels
        return tokenized


def _align_words_to_tags(
    word_ids: list[int | None],
    ner_tags: list[MultinerdTag],
) -> list[int]:
    label_ids = []
    previous_word_id = None
    for word_id in word_ids:
        if word_id is None:
            label_ids.append(-100)
        elif word_id != previous_word_id:
            label_ids.append(ner_tags[word_id])
        else:
            label_ids.append(-100)

        previous_word_id = word_id

    return label_ids


def _filter_en(batch: MultinerdBatch) -> list[bool]:
    return [lang == "en" for lang in batch["lang"]]
