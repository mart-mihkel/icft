"""MultiNERD dataset loading, prompting, and tokenization."""

from textwrap import dedent
from typing import TYPE_CHECKING, Literal, TypedDict, cast

from datasets.load import load_dataset
from datasets.utils.info_utils import VerificationMode

from saspbft.logging import logger
from saspbft.types import Architecture, DatasetInfo

if TYPE_CHECKING:
    from datasets.dataset_dict import DatasetDict
    from datasets.splits import Split
    from transformers import BatchEncoding, PreTrainedTokenizerFast

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
    "PER",
    "ORG",
    "LOC",
    "ANIM",
    "BIO",
    "CEL",
    "DIS",
    "EVE",
    "FOOD",
    "INST",
    "MEDIA",
    "MYTH",
    "PLANT",
    "TIME",
    "VEHI",
]


class MultinerdExamples(TypedDict):
    """A batch of raw MultiNERD examples."""

    tokens: list[list[str]]
    ner_tags: list[list[int]]
    lang: list[MultinerdLang]


_id2label_bio: dict[int, str] = {
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

id2label: dict[int, MultinerdTag] = {
    0: "PER",
    1: "ORG",
    2: "LOC",
    3: "ANIM",
    4: "BIO",
    5: "CEL",
    6: "DIS",
    7: "EVE",
    8: "FOOD",
    9: "INST",
    10: "MEDIA",
    11: "MYTH",
    12: "PLANT",
    13: "TIME",
    14: "VEHI",
}

label2id: dict[MultinerdTag, int] = {
    "PER": 0,
    "ORG": 1,
    "LOC": 2,
    "ANIM": 3,
    "BIO": 4,
    "CEL": 5,
    "DIS": 6,
    "EVE": 7,
    "FOOD": 8,
    "INST": 9,
    "MEDIA": 10,
    "MYTH": 11,
    "PLANT": 12,
    "TIME": 13,
    "VEHI": 14,
}

shots = [
    ("Sentence: John works at Google in California.\nEntity: John\nTag: PER\n"),
    ("Sentence: Paris is the capital of France.\nEntity: Paris\nTag: LOC\n"),
    ("Sentence: The dog chased the cat across the garden.\nEntity: dog\nTag: ANIM\n"),
    ("Sentence: I love eating sushi and pasta for dinner.\nEntity: sushi\nTag: FOOD\n"),
    ("Sentence: The car drove quickly down the highway.\nEntity: car\nTag: VEHI\n"),
    (
        "Sentence: The meeting was held at the United Nations headquarters.\n"
        "Entity: United Nations\n"
        "Tag: ORG\n"
    ),
    (
        "Sentence: Evolution shaped the diversity of life on Earth.\n"
        "Entity: Evolution\n"
        "Tag: BIO\n"
    ),
    (
        "Sentence: Einstein developed the theory of relativity.\n"
        "Entity: Einstein\n"
        "Tag: CEL\n"
    ),
    (
        "Sentence: The patient was diagnosed with diabetes last year.\n"
        "Entity: diabetes\n"
        "Tag: DIS\n"
    ),
    (
        "Sentence: The Olympics will be held in Tokyo next summer.\n"
        "Entity: Olympics\n"
        "Tag: EVE\n"
    ),
    (
        "Sentence: The telescope was invented several centuries ago.\n"
        "Entity: telescope\n"
        "Tag: INST\n"
    ),
    (
        "Sentence: I watched an interesting movie on Netflix last night.\n"
        "Entity: Netflix\n"
        "Tag: MEDIA\n"
    ),
    (
        "Sentence: The dragon guarded the ancient treasure in the cave.\n"
        "Entity: dragon\n"
        "Tag: MYTH\n"
    ),
    (
        "Sentence: Roses bloom beautifully in the garden during spring.\n"
        "Entity: Roses\n"
        "Tag: PLANT\n"
    ),
    (
        "Sentence: The meeting is scheduled for Monday morning.\n"
        "Entity: Monday\n"
        "Tag: TIME\n"
    ),
]


def _enc_sys_prompt(sep: str) -> str:
    return f"What is the NER tag of the entity in the sentence?{sep}"


def _enc_prompt(sentence: str, entity: str, sep: str) -> str:
    return f"{sentence}{sep}{entity}"


def _dec_sys_prompt() -> str:
    return dedent(f"""
        Identify the NER tag of the entity in the sentence.
        Possible tags are: {", ".join(id2label.values())}.

        Output only the tag.
    """).strip()


def _dec_prompt(sentence: str, entity: str) -> str:
    return dedent(f"""
        Sentence: {sentence}
        Entity: {entity}
        Tag:
    """).strip()


def _encdec_sys_prompt() -> str:
    return dedent(f"""
        ner: identify the ner tag of the entity in the sentence.
        tags: {", ".join(id2label.values())}.

        output only the tag.
    """).strip()


def _encdec_prompt(sentence: str, entity: str) -> str:
    return dedent(f"""
        sentence: {sentence}
        entity: {entity}
        tag:
    """).strip()


def _get_sys_prompt(tokenizer: PreTrainedTokenizerFast, arch: Architecture) -> str:
    if arch == "encoder":
        return _enc_sys_prompt(sep=tokenizer.sep_token)

    if arch == "decoder":
        return _dec_sys_prompt()

    if arch == "encoder-decoder":
        return _encdec_sys_prompt()


def _get_prompt(
    tokenizer: PreTrainedTokenizerFast,
    arch: Architecture,
    sentence: str,
    entity: str,
    n_shot: int,
) -> str:
    if arch == "encoder":
        prompt = _enc_prompt(sentence, entity, tokenizer.sep_token)
    elif arch == "decoder":
        prompt = _dec_prompt(sentence, entity)
    elif arch == "encoder-decoder":
        prompt = _encdec_prompt(sentence, entity)

    if n_shot > 0:
        if n_shot > len(shots):
            msg = "requested more examples than exist"
            raise ValueError(msg)
        prompt_shots = "\n".join(shots[:n_shot])
        prompt = f"{prompt_shots}\n{prompt}"

    return prompt


def _tokenize_batch(
    examples: MultinerdExamples,
    tokenizer: PreTrainedTokenizerFast,
    arch: Architecture,
    n_shot: int,
) -> dict[str, list]:
    all_ids, all_attn, all_tti, all_labels = [], [], [], []

    sys = _get_sys_prompt(tokenizer, arch)
    token_tags = zip(examples["tokens"], examples["ner_tags"], strict=True)
    for tokens, raw_tag_ids in token_tags:
        sentence = " ".join(tokens)
        entities, tag_ids = _join_spans(tokens, raw_tag_ids)

        for entity, tag_id in zip(entities, tag_ids, strict=True):
            if tag_id == -1:
                continue

            prompt = _get_prompt(tokenizer, arch, sentence, entity, n_shot)
            if tokenizer.chat_template is None:
                prompt_enc = tokenizer(
                    f"{sys}\n{prompt}",
                    truncation=True,
                    return_token_type_ids=True,
                )
            else:
                conv = [
                    {"role": "system", "content": sys},
                    {"role": "user", "content": prompt},
                ]

                prompt_enc = tokenizer.apply_chat_template(
                    conv,
                    truncation=True,
                    return_dict=True,
                    return_token_type_ids=True,
                    add_generation_prompt=arch != "encoder",
                )

            prompt_enc = cast("BatchEncoding", prompt_enc)
            prompt_len = len(cast("list[int]", prompt_enc["input_ids"]))

            if arch == "encoder":
                all_ids.append(prompt_enc["input_ids"])
                all_attn.append(prompt_enc["attention_mask"])
                all_tti.append(prompt_enc.get("token_type_ids"))
                all_labels.append(tag_id)
                continue

            if tokenizer.chat_template is None:
                answer = f"{sys}\n{prompt} {id2label[tag_id]}"
                answer_enc = tokenizer(
                    answer,
                    truncation=True,
                    return_token_type_ids=True,
                )
            else:
                conv = [
                    {"role": "system", "content": sys},
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": id2label[tag_id]},
                ]

                answer_enc = tokenizer.apply_chat_template(
                    conv,
                    truncation=True,
                    return_dict=True,
                    return_token_type_ids=True,
                )

            answer_enc = cast("BatchEncoding", answer_enc)
            labels_enc = cast("list[int]", answer_enc["input_ids"]).copy()

            if arch == "decoder":
                all_ids.append(answer_enc["input_ids"])
                all_attn.append(answer_enc["attention_mask"])
                all_tti.append(answer_enc.get("token_type_ids"))
                labels_enc[:prompt_len] = [-100] * prompt_len
                all_labels.append(labels_enc)
                continue

            if arch == "encoder-decoder":
                all_ids.append(prompt_enc["input_ids"])
                all_attn.append(prompt_enc["attention_mask"])
                all_tti.append(prompt_enc.get("token_type_ids"))
                idx = prompt_len - int(labels_enc[-1] == tokenizer.eos_token_id)
                all_labels.append(labels_enc[idx:])
                continue

    return {
        "input_ids": all_ids,
        "attention_mask": all_attn,
        "token_type_ids": all_tti,
        "labels": all_labels,
    }


def _join_spans(
    tokens: list[str],
    tag_ids: list[int],
) -> tuple[list[str], list[int]]:
    out_ids = []
    out_tokens = []
    for token, tag_id in zip(tokens, tag_ids, strict=True):
        tag = _id2label_bio[tag_id]

        if tag.startswith("B-"):
            tag = cast("MultinerdTag", tag[2:])
            out_ids.append(label2id[tag])
            out_tokens.append(token)
        elif tag.startswith("I-"):
            out_tokens[-1] = f"{out_tokens[-1]} {token}"
        elif tag == "O":
            out_ids.append(-1)
            out_tokens.append(token)
        else:
            tag = cast("MultinerdTag", tag)
            out_ids.append(label2id[tag])
            out_tokens.append(token)

    return out_tokens, out_ids


def _filter_english(batch: MultinerdExamples) -> list[bool]:
    return [lang == "en" for lang in batch["lang"]]


def load_multinerd(
    tokenizer: PreTrainedTokenizerFast,
    arch: Architecture,
    n_shot: int = 0,
    split: Split | None = None,
    *,
    filter_en: bool = True,
) -> tuple[DatasetDict, DatasetInfo]:
    """
    Initialize a modified subset of the MultiNERD dataset.

    The BIO tagging task is converted to a regular NER tagging task by joining
    tokens with B- and I- prefixes into a single span. O tags are dropped
    entirely.

    Each token is split into a separate sample containing the entire context
    sentence and the target token. The task is to classify the tag of the token
    in the entire sequence.
    """
    data = load_dataset(
        "Babelscape/multinerd",
        split=split,
        verification_mode=VerificationMode.NO_CHECKS,
    )

    data = cast("DatasetDict", data)

    if filter_en:
        logger.warning("using english only subset")
        data = data.filter(_filter_english, batched=True)

    logger.debug("tokenize multinerd")
    cols = ["tokens", "ner_tags", "lang"]
    fn_kwargs = {"tokenizer": tokenizer, "n_shot": n_shot, "arch": arch}
    data = data.map(
        _tokenize_batch,
        num_proc=4,
        batched=True,
        remove_columns=cols,
        fn_kwargs=fn_kwargs,
    )

    for subsplit in data:
        logger.debug("tokenized %d %s samples", len(data[subsplit]), subsplit)

    info = DatasetInfo(
        id2label=cast("dict[int, str]", id2label),
        label2id=cast("dict[str, int]", label2id),
        system_prompt=_get_sys_prompt(tokenizer, arch),
    )

    return data, info
