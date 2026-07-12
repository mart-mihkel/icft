from textwrap import dedent
from typing import Literal, TypedDict, cast

from datasets.dataset_dict import DatasetDict
from datasets.load import load_dataset
from datasets.splits import Split
from transformers import BatchEncoding, PreTrainedTokenizerFast

from instruct.logging import logger
from instruct.types import Architecture, DatasetInfo

type BoolQALabel = Literal["no", "yes"]


class BoolqExample(TypedDict):
    idx: int
    passage: str
    question: str
    label: int


id2label: dict[int, BoolQALabel] = {0: "no", 1: "yes"}
label2id: dict[BoolQALabel, int] = {"no": 0, "yes": 1}

shots = [
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


def _enc_prompt(example: BoolqExample, sep: str) -> str:
    return f"{example['question']}{sep}{example['passage']}"


def _dec_sys_prompt() -> str:
    return dedent("""
        Answer the question based on the passage.
        Do not provide any explanation.
        Answer with only yes or no
    """).strip()


def _dec_prompt(example: BoolqExample) -> str:
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


def _encdec_prompt(example: BoolqExample) -> str:
    return dedent(f"""
        passage: {example["passage"]}
        question: {example["question"]}
        answer (yes/no):
    """).strip()


def _get_sys_prompt(
    tokenizer: PreTrainedTokenizerFast,
    arch: Architecture,
) -> str:
    if arch == "encoder":
        return _enc_sys_prompt(sep=tokenizer.sep_token)

    if arch == "decoder":
        return _dec_sys_prompt()

    if arch == "encoder-decoder":
        return _encdec_sys_prompt()


def _get_prompt(
    tokenizer: PreTrainedTokenizerFast,
    arch: Architecture,
    example: BoolqExample,
    n_shot: int,
) -> str:
    if arch == "encoder":
        prompt = _enc_prompt(example, tokenizer.sep_token)
    elif arch == "decoder":
        prompt = _dec_prompt(example)
    elif arch == "encoder-decoder":
        prompt = _encdec_prompt(example)

    if n_shot > 0:
        assert n_shot <= len(shots), "requested more examples than exist"
        prompt_shots = "\n".join(shots[:n_shot])
        prompt = f"{prompt_shots}\n{prompt}"

    return prompt


def _tokenize(
    example: BoolqExample,
    tokenizer: PreTrainedTokenizerFast,
    arch: Architecture,
    n_shot: int,
) -> BatchEncoding:
    _id2label = id2label | {-1: "private"}

    sys = _get_sys_prompt(tokenizer, arch)
    prompt = _get_prompt(tokenizer, arch, example, n_shot)
    label_id = example["label"]
    label = _id2label[label_id]

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

    prompt_enc = cast(BatchEncoding, prompt_enc)
    prompt_len = len(cast(list[int], prompt_enc["input_ids"]))

    if arch == "encoder":
        prompt_enc["label"] = label_id
        return prompt_enc

    if tokenizer.chat_template is None:
        answer = f"{sys}\n{prompt} {label}"
        answer_enc = tokenizer(
            answer,
            truncation=True,
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
            return_dict=True,
            return_token_type_ids=True,
        )

    answer_enc = cast(BatchEncoding, answer_enc)
    labels_enc = cast(list[int], answer_enc["input_ids"]).copy()

    if arch == "decoder":
        labels_enc[:prompt_len] = [-100] * prompt_len
        answer_enc["labels"] = labels_enc
        return answer_enc

    if arch == "encoder-decoder":
        idx = prompt_len - int(labels_enc[-1] == tokenizer.eos_token_id)
        prompt_enc["labels"] = labels_enc[idx:]
        return prompt_enc


def load_boolq(
    tokenizer: PreTrainedTokenizerFast,
    arch: Architecture,
    n_shot: int = 0,
    split: Split | None = None,
) -> tuple[DatasetDict, DatasetInfo]:
    data = cast(DatasetDict, load_dataset("aps/super_glue", "boolq", split=split))

    if "validation" in data:
        logger.debug("rename 'validation' to 'dev'")
        data["dev"] = data.pop("validation")

    logger.debug("tokenize boolq")
    cols = ["question", "passage", "label"]
    fn_kwargs = {"tokenizer": tokenizer, "n_shot": n_shot, "arch": arch}
    data = data.map(_tokenize, remove_columns=cols, fn_kwargs=fn_kwargs)
    for subsplit in data:
        logger.debug("tokenized %d %s samples", len(data[subsplit]), subsplit)

    info = DatasetInfo(
        id2label=cast(dict[int, str], id2label),
        label2id=cast(dict[str, int], label2id),
        system_prompt=_get_sys_prompt(tokenizer, arch),
    )

    return data, info
