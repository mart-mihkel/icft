"""EstNER dataset loading, prompting, and tokenization."""

from textwrap import dedent
from typing import TYPE_CHECKING, Literal, TypedDict, cast

from datasets.load import load_dataset

from saspbft.datasets.truncation import get_max_length
from saspbft.logging import logger
from saspbft.types import Architecture, DatasetInfo

if TYPE_CHECKING:
    from datasets.arrow_dataset import Dataset
    from datasets.dataset_dict import DatasetDict
    from datasets.splits import Split
    from transformers import BatchEncoding, PreTrainedTokenizerFast

type _EstnerTag = Literal[
    "O",
    "PER",
    "GPE",
    "LOC",
    "ORG",
    "PROD",
    "EVENT",
    "DATE",
    "TIME",
    "TITLE",
    "MONEY",
    "PERCENT",
]


class _EstnerExamples(TypedDict):
    """A batch of raw EstNER examples."""

    doc_id: list[int]
    sent_id: list[int]
    tokens: list[list[str]]
    ner_tags: list[list[str]]
    ner_tags1: list[list[str]]
    ner_tags2: list[list[str]]


_ID2LABEL_BIO: dict[int, str] = {
    0: "O",
    1: "B-PER",
    2: "I-PER",
    3: "B-GPE",
    4: "I-GPE",
    5: "B-LOC",
    6: "I-LOC",
    7: "B-ORG",
    8: "I-ORG",
    9: "B-PROD",
    10: "I-PROD",
    11: "B-EVENT",
    12: "I-EVENT",
    13: "B-DATE",
    14: "I-DATE",
    15: "B-TIME",
    16: "I-TIME",
    17: "B-TITLE",
    18: "I-TITLE",
    19: "B-MONEY",
    20: "I-MONEY",
    21: "B-PERCENT",
    22: "I-PERCENT",
}

_ID2LABEL: dict[int, _EstnerTag] = {
    0: "O",
    1: "PER",
    2: "GPE",
    3: "LOC",
    4: "ORG",
    5: "PROD",
    6: "EVENT",
    7: "DATE",
    8: "TIME",
    9: "TITLE",
    10: "MONEY",
    11: "PERCENT",
}

_LABEL2ID: dict[_EstnerTag, int] = {
    "O": 0,
    "PER": 1,
    "GPE": 2,
    "LOC": 3,
    "ORG": 4,
    "PROD": 5,
    "EVENT": 6,
    "DATE": 7,
    "TIME": 8,
    "TITLE": 9,
    "MONEY": 10,
    "PERCENT": 11,
}

_COLS = ["doc_id", "sent_id", "tokens", "ner_tags", "ner_tags_2", "ner_tags_3"]


def _format_shot(sentence: str, entity: str, tag: str) -> str:
    return f"Lause: {sentence}\nNimeüksus: {entity}\nMärgend: {tag}\n"


def _enc_sys_prompt(sep: str) -> str:
    return f"Mis on nimeüksuse NER märgen lauses?{sep}"


def _enc_prompt(sentence: str, entity: str, sep: str) -> str:
    return f"{sentence}{sep}{entity}"


def _dec_sys_prompt() -> str:
    return dedent(f"""
        Määra nimeüksuse NER märgen lauses.
        Võimalikut märgendid on: {", ".join(_ID2LABEL.values())}.

        Vasta ainult märgendiga.
    """).strip()


def _dec_prompt(sentence: str, entity: str) -> str:
    return dedent(f"""
        Lause: {sentence}
        Nimeüksus: {entity}
        Märgend:
    """).strip()


def _encdec_sys_prompt() -> str:
    return dedent(f"""
        ner: määra nimeüksuse NER märgen lauses.
        märgendid: {", ".join(_ID2LABEL.values())}.

        vasta ainult märgendiga.
    """).strip()


def _encdec_prompt(sentence: str, entity: str) -> str:
    return dedent(f"""
        lause: {sentence}
        nimeüksus: {entity}
        märgend:
    """).strip()


def get_sys_prompt(tokenizer: PreTrainedTokenizerFast, arch: Architecture) -> str:
    """Return the system prompt for the given architecture."""
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
    shots: list[str],
) -> str:
    if arch == "encoder":
        prompt = _enc_prompt(sentence, entity, tokenizer.sep_token)
    elif arch == "decoder":
        prompt = _dec_prompt(sentence, entity)
    elif arch == "encoder-decoder":
        prompt = _encdec_prompt(sentence, entity)

    if shots:
        prompt_shots = "\n".join(shots)
        prompt = f"{prompt_shots}\n{prompt}"

    return prompt


def _tokenize_batch(
    examples: _EstnerExamples,
    tokenizer: PreTrainedTokenizerFast,
    arch: Architecture,
    shots: list[str],
    num_virtual_tokens: int = 0,
) -> dict[str, list]:
    all_ids, all_attn, all_tti, all_labels, all_truncated = [], [], [], [], []
    max_length = get_max_length(tokenizer, num_virtual_tokens)

    sys = get_sys_prompt(tokenizer, arch)
    for tokens, raw_tags in zip(examples["tokens"], examples["ner_tags"], strict=True):
        sentence = " ".join(tokens)
        entities, tags = _join_spans(tokens, raw_tags)

        for entity, tag in zip(entities, tags, strict=True):
            prompt = _get_prompt(tokenizer, arch, sentence, entity, shots)
            if tokenizer.chat_template is None:
                prompt_enc = tokenizer(
                    f"{sys}\n{prompt}",
                    truncation=True,
                    max_length=max_length,
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
                    max_length=max_length,
                    return_dict=True,
                    return_token_type_ids=True,
                    add_generation_prompt=arch != "encoder",
                )

            prompt_enc = cast("BatchEncoding", prompt_enc)
            prompt_len = len(cast("list[int]", prompt_enc["input_ids"]))
            truncated = max_length is not None and prompt_len >= max_length

            if arch == "encoder":
                all_ids.append(prompt_enc["input_ids"])
                all_attn.append(prompt_enc["attention_mask"])
                all_tti.append(prompt_enc.get("token_type_ids"))
                all_labels.append(_LABEL2ID[tag])
                all_truncated.append(truncated)
                continue

            if tokenizer.chat_template is None:
                answer = f"{sys}\n{prompt} {tag}"
                answer_enc = tokenizer(
                    answer,
                    truncation=True,
                    max_length=max_length,
                    return_token_type_ids=True,
                )
            else:
                conv = [
                    {"role": "system", "content": sys},
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": tag},
                ]

                answer_enc = tokenizer.apply_chat_template(
                    conv,
                    truncation=True,
                    max_length=max_length,
                    return_dict=True,
                    return_token_type_ids=True,
                )

            answer_enc = cast("BatchEncoding", answer_enc)
            labels_enc = cast("list[int]", answer_enc["input_ids"]).copy()
            truncated = truncated or (
                max_length is not None and len(labels_enc) >= max_length
            )

            if arch == "decoder":
                all_ids.append(answer_enc["input_ids"])
                all_attn.append(answer_enc["attention_mask"])
                all_tti.append(answer_enc.get("token_type_ids"))
                labels_enc[:prompt_len] = [-100] * prompt_len
                all_labels.append(labels_enc)
                all_truncated.append(truncated)
                continue

            if arch == "encoder-decoder":
                all_ids.append(prompt_enc["input_ids"])
                all_attn.append(prompt_enc["attention_mask"])
                all_tti.append(prompt_enc.get("token_type_ids"))
                idx = prompt_len - int(labels_enc[-1] == tokenizer.eos_token_id)
                all_labels.append(labels_enc[idx:])
                all_truncated.append(truncated)
                continue

    return {
        "input_ids": all_ids,
        "attention_mask": all_attn,
        "token_type_ids": all_tti,
        "labels": all_labels,
        "truncated": all_truncated,
    }


def _join_spans(
    tokens: list[str],
    tags: list[str],
) -> tuple[list[str], list[_EstnerTag]]:
    out_tags = []
    out_tokens = []
    for token, raw_tag in zip(tokens, tags, strict=True):
        if raw_tag.startswith("B-"):
            tag = cast("_EstnerTag", raw_tag[2:])
            out_tags.append(tag)
            out_tokens.append(token)
        elif raw_tag.startswith("I-"):
            out_tokens[-1] = f"{out_tokens[-1]} {token}"
        else:
            tag = cast("_EstnerTag", raw_tag)
            out_tags.append(tag)
            out_tokens.append(token)

    return out_tokens, out_tags


def _sample_shots(data: Dataset, n_shot: int) -> list[str]:
    if n_shot == 0:
        return []

    shots = []
    for example in data:
        tokens = cast("list[str]", example["tokens"])
        raw_tags = cast("list[str]", example["ner_tags"])
        sentence = " ".join(tokens)
        entities, tags = _join_spans(tokens, raw_tags)
        if entities:
            shots.append(_format_shot(sentence, entities[0], tags[0]))

        if len(shots) == n_shot:
            return shots

    msg = f"requested more than {len(shots)} examples"
    raise ValueError(msg)


def load_estner(
    tokenizer: PreTrainedTokenizerFast,
    arch: Architecture,
    *,
    n_shot: int = 0,
    num_virtual_tokens: int = 0,
    split: Split | None = None,
) -> tuple[DatasetDict, DatasetInfo]:
    """
    Initialize a modified version of the EstNER dataset.

    The BIO tagging task is converted to a regular NER tagging task by joining
    tokens with B- and I- prefixes into a single span.

    Each token is split into a separate sample containing the entire context
    sentence and the target token. The task is to classify the tag of the token
    in the entire sequence.
    """
    data = cast("DatasetDict", load_dataset("tartuNLP/EstNER", split=split))

    if "dev" in data:
        logger.debug("rename 'dev' to 'validation'")
        data["validation"] = data.pop("dev")

    shots = _sample_shots(data["train"], n_shot)

    logger.debug("tokenize estner")
    fn_kwargs = {
        "arch": arch,
        "shots": shots,
        "tokenizer": tokenizer,
        "num_virtual_tokens": num_virtual_tokens,
    }

    data = data.map(
        _tokenize_batch,
        batched=True,
        remove_columns=_COLS,
        fn_kwargs=fn_kwargs,
    )

    for subsplit in data:
        logger.debug("tokenized %d %s samples", len(data[subsplit]), subsplit)

    info = DatasetInfo(
        id2label=cast("dict[int, str]", _ID2LABEL),
        label2id=cast("dict[str, int]", _LABEL2ID),
        system_prompt=get_sys_prompt(tokenizer, arch),
    )

    return data, info
