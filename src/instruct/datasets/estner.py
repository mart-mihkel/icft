"""EstNER dataset loading, prompting, and tokenization."""

from textwrap import dedent
from typing import TYPE_CHECKING, Literal, TypedDict, cast

from datasets.load import load_dataset

from instruct.logging import logger
from instruct.types import Architecture, DatasetInfo

if TYPE_CHECKING:
    from datasets.dataset_dict import DatasetDict
    from datasets.splits import Split
    from transformers import BatchEncoding, PreTrainedTokenizerFast

type EstnerTag = Literal[
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


class EstnerExamples(TypedDict):
    """A batch of raw EstNER examples."""

    doc_id: list[int]
    sent_id: list[int]
    tokens: list[list[str]]
    ner_tags: list[list[str]]
    ner_tags1: list[list[str]]
    ner_tags2: list[list[str]]


_id2label_full: dict[int, str] = {
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

id2label: dict[int, EstnerTag] = {
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

label2id: dict[EstnerTag, int] = {
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

examples = [
    ("Lause: Mari töötab Google'is Californias.\nNimeüksus: Mari\nMärgend: PER\n"),
    ("Lause: Koosolek toimus ÜRO peakorteris.\nNimeüksus: ÜRO\nMärgend: ORG\n"),
    ("Lause: Tallinn on Eesti pealinn.\nNimeüksus: Tallinn\nMärgend: LOC\n"),
    ("Lause: Eesti asub Põhja-Euroopas.\nNimeüksus: Eesti\nMärgend: GPE\n"),
    ("Lause: Ta jõi hommikul kuuma kohvi.\nNimeüksus: kohvi\nMärgend: PROD\n"),
    ("Lause: Võidupüha tähistatakse juunis.\nNimeüksus: Võidupüha\nMärgend: EVENT\n"),
    ("Lause: Ta sündis 1990. aastal.\nNimeüksus: 1990. aastal\nMärgend: DATE\n"),
    ("Lause: Tulemus näitab 75% kasvu.\nNimeüksus: 75%\nMärgend: PERCENT\n"),
    (
        "Lause: Koosolek algab kell kolm pärastlõunal.\n"
        "Nimeüksus: kell kolm\n"
        "Märgend: TIME\n"
    ),
    (
        "Lause: Ta kirjutas raamatu pealkirjaga 'Tarkus'.\n"
        "Nimeüksus: Tarkus\n"
        "Märgend: TITLE\n"
    ),
    (
        "Lause: Ta ostis uue auto 25000 euro eest.\n"
        "Nimeüksus: 25000 euro\n"
        "Märgend: MONEY\n"
    ),
]


def _enc_sys_prompt(sep: str) -> str:
    return f"Mis on nimeüksuse NER märgen lauses?{sep}"


def _enc_prompt(sentence: str, entity: str, sep: str) -> str:
    return f"{sentence}{sep}{entity}"


def _dec_sys_prompt() -> str:
    return dedent(f"""
        Määra nimeüksuse NER märgen lauses.
        Võimalikut märgendid on: {", ".join(id2label.values())}.

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
        märgendid: {", ".join(id2label.values())}.

        vasta ainult märgendiga.
    """).strip()


def _encdec_prompt(sentence: str, entity: str) -> str:
    return dedent(f"""
        lause: {sentence}
        nimeüksus: {entity}
        märgend:
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
        if n_shot > len(examples):
            msg = "requested more examples than exist"
            raise ValueError(msg)
        prompt_shots = "\n".join(examples[:n_shot])
        prompt = f"{prompt_shots}\n{prompt}"

    return prompt


def _tokenize_batch(
    examples: EstnerExamples,
    tokenizer: PreTrainedTokenizerFast,
    arch: Architecture,
    n_shot: int,
) -> dict[str, list]:
    all_ids, all_attn, all_tti, all_labels = [], [], [], []

    sys = _get_sys_prompt(tokenizer, arch)
    for tokens, raw_tags in zip(examples["tokens"], examples["ner_tags"], strict=True):
        sentence = " ".join(tokens)
        entities, tags = _join_spans(tokens, raw_tags)

        for entity, tag in zip(entities, tags, strict=True):
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
                all_labels.append(label2id[tag])
                continue

            if tokenizer.chat_template is None:
                answer = f"{sys}\n{prompt} {tag}"
                answer_enc = tokenizer(
                    answer,
                    truncation=True,
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
    tags: list[str],
) -> tuple[list[str], list[EstnerTag]]:
    out_tags = []
    out_tokens = []
    for token, raw_tag in zip(tokens, tags, strict=True):
        if raw_tag.startswith("B-"):
            tag = cast("EstnerTag", raw_tag[2:])
            out_tags.append(tag)
            out_tokens.append(token)
        elif raw_tag.startswith("I-"):
            out_tokens[-1] = f"{out_tokens[-1]} {token}"
        else:
            tag = cast("EstnerTag", raw_tag)
            out_tags.append(tag)
            out_tokens.append(token)

    return out_tokens, out_tags


def load_estner(
    tokenizer: PreTrainedTokenizerFast,
    arch: Architecture,
    n_shot: int = 0,
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

    logger.debug("tokenize estner")
    cols = ["doc_id", "sent_id", "tokens", "ner_tags", "ner_tags_2", "ner_tags_3"]
    fn_kwargs = {"tokenizer": tokenizer, "n_shot": n_shot, "arch": arch}
    data = data.map(
        _tokenize_batch,
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
