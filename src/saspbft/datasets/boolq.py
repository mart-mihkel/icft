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
_SHOTS = [
    (
        "Passage: The sky appears blue during the day due to Rayleigh scattering.\n"
        "Question: Is the sky blue?\n"
        "Answer: yes\n"
    ),
    (
        "Passage: Fish are animals that live exclusively underwater and breathe "
        "using gills.\n"
        "Question: Can fish breathe on land?\n"
        "Answer: no\n"
    ),
    (
        "Passage: Water freezes at 0 degrees Celsius and boils at 100 degrees "
        "Celsius at sea level.\n"
        "Question: Does water freeze at room temperature?\n"
        "Answer: no\n"
    ),
    (
        "Passage: The Earth orbits around the Sun in approximately 365 days.\n"
        "Question: Does the Earth orbit the Sun?\n"
        "Answer: yes\n"
    ),
    (
        "Passage: Photosynthesis is the process by which plants convert sunlight "
        "into energy.\n"
        "Question: Do plants produce their own food?\n"
        "Answer: yes\n"
    ),
    (
        "Passage: The Great Wall of China is visible from space with naked eye.\n"
        "Question: Is the Great Wall visible from space?\n"
        "Answer: no\n"
    ),
    (
        "Passage: Lightning is a discharge of electricity that occurs during "
        "thunderstorms.\n"
        "Question: Is lightning caused by electricity?\n"
        "Answer: yes\n"
    ),
    (
        "Passage: The human body contains 206 bones in adulthood.\n"
        "Question: Do adults have more than 300 bones?\n"
        "Answer: no\n"
    ),
]


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
    n_shot: int,
) -> str:
    if arch == "encoder":
        prompt = _enc_prompt(example, tokenizer.sep_token)
    elif arch == "decoder":
        prompt = _dec_prompt(example)
    elif arch == "encoder-decoder":
        prompt = _encdec_prompt(example)

    if n_shot > 0:
        if n_shot > len(_SHOTS):
            msg = "requested more examples than exist"
            raise ValueError(msg)
        prompt_shots = "\n".join(_SHOTS[:n_shot])
        prompt = f"{prompt_shots}\n{prompt}"

    return prompt


def _tokenize(
    example: _BoolqExample,
    tokenizer: PreTrainedTokenizerFast,
    arch: Architecture,
    n_shot: int,
    num_virtual_tokens: int = 0,
) -> BatchEncoding:
    _id2label = _ID2LABEL | {-1: "private"}
    max_length = get_max_length(tokenizer, num_virtual_tokens)

    sys = get_sys_prompt(tokenizer, arch)
    prompt = _get_prompt(tokenizer, arch, example, n_shot)
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
    num_virtual_tokens: int = 0,
    split: Split | None = None,
) -> tuple[DatasetDict, DatasetInfo]:
    """Load, tokenize, and prompt-format the BoolQ dataset."""
    data = cast("DatasetDict", load_dataset("aps/super_glue", "boolq", split=split))

    logger.debug("tokenize boolq")
    fn_kwargs = {
        "arch": arch,
        "n_shot": n_shot,
        "tokenizer": tokenizer,
        "num_virtual_tokens": num_virtual_tokens,
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
