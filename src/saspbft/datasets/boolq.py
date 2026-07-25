"""BoolQ dataset loading, prompting, and tokenization."""

from textwrap import dedent
from typing import TYPE_CHECKING, Literal, TypedDict, cast

from datasets.load import load_dataset

from saspbft.datasets.truncation import get_max_length
from saspbft.logging import logger
from saspbft.types import Architecture, DatasetInfo

if TYPE_CHECKING:
    from datasets.dataset_dict import DatasetDict
    from datasets.splits import Split
    from transformers import BatchEncoding, PreTrainedTokenizerFast

type _BoolqLabel = Literal["no", "yes"]


class _BoolqExample(TypedDict):
    """A single raw BoolQ example."""

    idx: int
    passage: str
    question: str
    label: int


_ID2LABEL: dict[int, _BoolqLabel] = {0: "no", 1: "yes"}
_LABEL2ID: dict[_BoolqLabel, int] = {"no": 0, "yes": 1}

_COLS = ["question", "passage", "label"]


def _format_shot(example: _BoolqExample) -> str:
    label = _ID2LABEL[example["label"]]
    return (
        f"Passage: {example['passage']}\n"
        f"Question: {example['question']}\n"
        f"Answer: {label}\n"
    )


def _enc_sys_prompt(sep: str) -> str:
    return f"Answer the question based on the passage.{sep}"


def _enc_prompt(example: _BoolqExample, sep: str) -> str:
    return f"{example['question']}{sep}{example['passage']}"


def _dec_sys_prompt() -> str:
    return dedent("""
        Answer the question based on the passage.
        Do not provide any explanation.
        Answer with only yes or no
    """).strip()


def _dec_prompt(example: _BoolqExample) -> str:
    return dedent(f"""
        Passage: {example["passage"]}
        Question: {example["question"]}
        Answer (yes/no):
    """).strip()


def _encdec_sys_prompt() -> str:
    return dedent("""
        boolqa: answer the question based on the passage
        answers: yes, no
        output only the answer
    """)


def _encdec_prompt(example: _BoolqExample) -> str:
    return dedent(f"""
        passage: {example["passage"]}
        question: {example["question"]}
        answer (yes/no):
    """).strip()


def get_sys_prompt(
    tokenizer: PreTrainedTokenizerFast,
    arch: Architecture,
) -> str:
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
    example: _BoolqExample,
    shots: list[str],
) -> str:
    if arch == "encoder":
        prompt = _enc_prompt(example, tokenizer.sep_token)
    elif arch == "decoder":
        prompt = _dec_prompt(example)
    elif arch == "encoder-decoder":
        prompt = _encdec_prompt(example)

    if shots:
        prompt_shots = "\n".join(shots)
        prompt = f"{prompt_shots}\n{prompt}"

    return prompt


def _tokenize(
    example: _BoolqExample,
    tokenizer: PreTrainedTokenizerFast,
    arch: Architecture,
    shots: list[str],
    n_virtual: int = 0,
) -> BatchEncoding:
    _id2label = _ID2LABEL | {-1: "private"}
    max_length = get_max_length(tokenizer, n_virtual)

    sys = get_sys_prompt(tokenizer, arch)
    prompt = _get_prompt(tokenizer, arch, example, shots)
    label_id = example["label"]
    label = _id2label[label_id]

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
        prompt_enc["label"] = label_id
        prompt_enc["truncated"] = truncated
        return prompt_enc

    if tokenizer.chat_template is None:
        answer = f"{sys}\n{prompt} {label}"
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
            {"role": "assistant", "content": label},
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
    truncated = truncated or (max_length is not None and len(labels_enc) >= max_length)

    if arch == "decoder":
        labels_enc[:prompt_len] = [-100] * prompt_len
        answer_enc["labels"] = labels_enc
        answer_enc["truncated"] = truncated
        return answer_enc

    if arch == "encoder-decoder":
        idx = prompt_len - int(labels_enc[-1] == tokenizer.eos_token_id)
        prompt_enc["labels"] = labels_enc[idx:]
        prompt_enc["truncated"] = truncated
        return prompt_enc


def load_boolq(
    tokenizer: PreTrainedTokenizerFast,
    arch: Architecture,
    *,
    n_shot: int = 0,
    n_virtual: int = 0,
    split: Split | None = None,
) -> tuple[DatasetDict, DatasetInfo]:
    """Load, tokenize, and prompt-format the BoolQ dataset."""
    data = cast("DatasetDict", load_dataset("aps/super_glue", "boolq", split=split))

    max_shots = len(data["train"])
    if n_shot > max_shots:
        msg = f"requested more than {max_shots} examples"
        raise ValueError(msg)
    elif n_shot > 0:
        sampled = data["train"].select(range(n_shot))
        shots = [_format_shot(s) for s in sampled]
    else:
        shots = []

    logger.debug("tokenize boolq")
    fn_kwargs = {
        "arch": arch,
        "shots": shots,
        "tokenizer": tokenizer,
        "n_virtual": n_virtual,
    }

    data = data.map(_tokenize, remove_columns=_COLS, fn_kwargs=fn_kwargs)
    for subsplit in data:
        logger.debug("tokenized %d %s samples", len(data[subsplit]), subsplit)

    info = DatasetInfo(
        id2label=cast("dict[int, str]", _ID2LABEL),
        label2id=cast("dict[str, int]", _LABEL2ID),
        system_prompt=get_sys_prompt(tokenizer, arch),
    )

    return data, info
