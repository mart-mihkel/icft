import logging
import random
import string
from typing import Literal, TypedDict, cast

import numpy as np
from datasets.arrow_dataset import Dataset
from datasets.load import load_dataset
from datasets.utils.info_utils import VerificationMode
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import BatchEncoding, PreTrainedTokenizer
from transformers.trainer_utils import EvalPrediction

logger = logging.getLogger(__name__)

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


class MultinerdMetrics(TypedDict):
    accuracy: float
    precision: float
    recall: float
    f1: float


class MultinerdBatch(TypedDict):
    tokens: list[list[str]]
    ner_tags: list[list[MultinerdTag]]
    lang: list[MultinerdLang]


class Multinerd:
    TAG2ID: dict[MultinerdTag, int] = {
        "O": 0,
        "B-PER": 1,
        "I-PER": 2,
        "B-ORG": 3,
        "I-ORG": 4,
        "B-LOC": 5,
        "I-LOC": 6,
        "B-ANIM": 7,
        "I-ANIM": 8,
        "B-BIO": 9,
        "I-BIO": 10,
        "B-CEL": 11,
        "I-CEL": 12,
        "B-DIS": 13,
        "I-DIS": 14,
        "B-EVE": 15,
        "I-EVE": 16,
        "B-FOOD": 17,
        "I-FOOD": 18,
        "B-INST": 19,
        "I-INST": 20,
        "B-MEDIA": 21,
        "I-MEDIA": 22,
        "B-MYTH": 23,
        "I-MYTH": 24,
        "B-PLANT": 25,
        "I-PLANT": 26,
        "B-TIME": 27,
        "I-TIME": 28,
        "B-VEHI": 29,
        "I-VEHI": 30,
    }

    ID2TAG: dict[int, MultinerdTag] = {
        0: "O",
        1: "B-PER",
        2: "I-PER",
        3: "B-ORG",
        4: "I-ORG",
        5: "B-LOC",
        6: "I-LOC",
        7: "B-ANIM",
        8: "I-ANIM",
        9: "B-BIO",
        10: "I-BIO",
        11: "B-CEL",
        12: "I-CEL",
        13: "B-DIS",
        14: "I-DIS",
        15: "B-EVE",
        16: "I-EVE",
        17: "B-FOOD",
        18: "I-FOOD",
        19: "B-INST",
        20: "I-INST",
        21: "B-MEDIA",
        22: "I-MEDIA",
        23: "B-MYTH",
        24: "I-MYTH",
        25: "B-PLANT",
        26: "I-PLANT",
        27: "B-TIME",
        28: "I-TIME",
        29: "B-VEHI",
        30: "I-VEHI",
    }

    _SYSTEM_TOKENS = """You are a named entity recognition model .
Given a target word and a sentence containing that word , predict the NER tag of the target word based on its context .

Example
Word : Paris
Sentence : Paris is the capital of France .
Output : B-LOC
""".split()

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        system_prompt: Literal["ner", "random", "none"],
        train_split: str,
        eval_split: str,
        test_split: str,
    ) -> None:
        logger.info("load multinerd")
        train, eval, test = load_dataset(
            "Babelscape/multinerd",
            split=[train_split, eval_split, test_split],
            verification_mode=VerificationMode.NO_CHECKS,
        )

        assert isinstance(train, Dataset)
        assert isinstance(eval, Dataset)
        assert isinstance(test, Dataset)

        logger.info("filter english")
        train = train.filter(_filter_english, batched=True)
        eval = eval.filter(_filter_english, batched=True)
        eval = eval.filter(_filter_english, batched=True)

        logger.info("tokenize")
        sep_token = tokenizer.special_tokens_map["sep_token"]
        assert isinstance(sep_token, str), "Expected one separator token"

        self.tokenizer = tokenizer
        self.sep_token = sep_token

        self.system_tokens = []
        if system_prompt == "ner":
            self.system_tokens = self._SYSTEM_TOKENS
        elif system_prompt == "random":
            self.system_tokens = _random_tokens(len(self._SYSTEM_TOKENS))

        self.train = train.map(
            self._tokenize, batched=True, remove_columns=train.column_names
        )
        self.eval = eval.map(
            self._tokenize, batched=True, remove_columns=eval.column_names
        )
        self.test = test.map(
            self._tokenize, batched=True, remove_columns=test.column_names
        )

    def _tokenize(self, batch: MultinerdBatch) -> BatchEncoding:
        labels: list[MultinerdTag] = []
        prompts: list[list[str]] = []
        for tokens, tags in zip(batch["tokens"], batch["ner_tags"]):
            for token, tag in zip(tokens, tags):
                prompt = self._build_prompt(target=token, sentence=tokens)
                prompts.append(prompt)
                labels.append(tag)

        tokenized = self.tokenizer(prompts, truncation=True, is_split_into_words=True)
        tokenized["labels"] = labels
        return tokenized

    def _build_prompt(self, target: str, sentence: list[str]) -> list[str]:
        return (
            self.system_tokens
            + ([self.sep_token] if self.system_tokens else [])
            + [target, self.sep_token]
            + sentence
        )

    @staticmethod
    def compute_metrics(eval_pred: EvalPrediction) -> dict[str, float]:
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        metrics = MultinerdMetrics(
            accuracy=accuracy_score(labels, preds),
            precision=precision_score(
                labels, preds, average="macro", zero_division=np.nan
            ),
            recall=recall_score(labels, preds, average="macro"),
            f1=f1_score(labels, preds, average="macro"),
        )

        return cast(dict[str, float], metrics)


def _random_tokens(n: int, min_len: int = 3, max_len: int = 12) -> list[str]:
    chars = string.ascii_lowercase
    words = []
    for _ in range(n):
        k = random.randint(min_len, max_len)
        word = "".join(random.choices(chars, k=k))
        words.append(word)

    return words


def _filter_english(batch: MultinerdBatch) -> list[bool]:
    return [lang == "en" for lang in batch["lang"]]
