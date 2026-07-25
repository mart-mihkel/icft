"""OBL (oblique phrase removal) dataset loading, prompting, and tokenization."""

from textwrap import dedent
from typing import TYPE_CHECKING, Literal, TypedDict, cast

from datasets.arrow_dataset import Dataset
from datasets.dataset_dict import DatasetDict

from saspbft.datasets.truncation import get_max_length
from saspbft.logging import logger
from saspbft.types import Architecture, DatasetInfo

if TYPE_CHECKING:
    from datasets.splits import Split
    from transformers import BatchEncoding, PreTrainedTokenizerFast

_PERMALINK = (
    "https://raw.githubusercontent.com/estnltk/estnltk-model-data/"
    "c6f373a49d29a5a4b5e92326e5309b173bfcd3d1"
    "/phrase_removal/benchmarks/obl_phrases/obl_max/data/obl_all_hand_annotated.csv"
)

type _OblLabel = Literal["seotud", "vaba", "ebaloomulik", "liigne koma", "kaheldav"]


class _OblExample(TypedDict):
    """A single raw OBL example."""

    id: int
    fpath: str
    sentence: str
    remove_start: int
    remove_end: int
    removed: str
    label: _OblLabel
    short_sent: str
    cons_score: float
    ual: float
    la: float


_EN_TO_ET: dict[str, _OblLabel] = {
    "bound": "seotud",
    "free": "vaba",
    "unnatural": "ebaloomulik",
    "redundant comma": "liigne koma",
    "dubious": "kaheldav",
}

_ID2LABEL: dict[int, _OblLabel] = {
    0: "seotud",
    1: "ebaloomulik",
    2: "liigne koma",
    3: "vaba",
    4: "kaheldav",
}

_LABEL2ID: dict[_OblLabel, int] = {
    "seotud": 0,
    "ebaloomulik": 1,
    "liigne koma": 2,
    "vaba": 3,
    "kaheldav": 4,
}

_COLS = [
    "id",
    "fpath",
    "sentence",
    "remove_start",
    "remove_end",
    "removed",
    "label",
    "short_sent",
    "cons_score",
    "ual",
    "la",
]


def _format_shot(example: _OblExample) -> str:
    return (
        f"Lause: {example['sentence']}\n"
        f"Fraas: {example['removed']}\n"
        f"Kategooria: {example['label']}\n"
    )


def _enc_sys_prompt(sep: str) -> str:
    return dedent(f"""
        Mis on seos fragmendi ja lause vahel,
        kas see on seotud, vaba, kaheldav, ebaloomulik või liigne koma?{sep}
    """).strip()


def _enc_prompt(example: _OblExample, sep: str) -> str:
    return f"{example['removed']}{sep}{example['sentence']}"


def _dec_sys_prompt() -> str:
    return dedent("""
        Sa oled liigse fraasi märgendamise mudel.
        Liigita eemaldatud fraas ühte kategooriasse:

        - liigne koma: ebavajalik või vale koma
        - ebaloomulik: grammatiliselt korrektne, kuid kohmakas või mittekeeleomane
        - kaheldav: ebaselge või piiripealne juhtum
        - seotud: vajalik grammatilise struktuuri jaoks, eemaldamine rikub grammatika
        - vaba: eemaldatav ilma tähendust või grammatikat muutmata

        Väljasta ainult kategooria.
    """).strip()


def _dec_prompt(example: _OblExample) -> str:
    return dedent(f"""
        Lause: {example["sentence"]}
        Fraas: {example["removed"]}
        Kategooria:
    """).strip()


def _encdec_sys_prompt() -> str:
    return dedent("""
        liigse fraasi märgendamine: liigita eemaldatud tekst ühte viiest kategooriast.

        liigne koma: ebavajalik või vale koma
        ebaloomulik: grammatiliselt valikuline, kuid kohmakas või mittekeeleomane
        kaheldav: ebaselge või piiripealne juhtum
        seotud: vajalik grammatilise struktuuri jaoks; eemaldamine rikub grammatika
        vaba: eemaldatav ilma tähendust või grammatikat muutmata

        väljasta ainult kategooria
    """).strip()


def _encdec_prompt(example: _OblExample) -> str:
    return dedent(f"""
        lause: {example["sentence"]}
        fraas: {example["removed"]}
        kategooria:
    """).strip()


def obl_sys_prompt(tokenizer: PreTrainedTokenizerFast, arch: Architecture) -> str:
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
    example: _OblExample,
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
    example: _OblExample,
    tokenizer: PreTrainedTokenizerFast,
    arch: Architecture,
    shots: list[str],
    n_virtual: int = 0,
) -> BatchEncoding:
    max_length = get_max_length(tokenizer, n_virtual)
    sys = obl_sys_prompt(tokenizer, arch)
    prompt = _get_prompt(tokenizer, arch, example, shots)
    label = example["label"]

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
        prompt_enc["label"] = _LABEL2ID[label]
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


def _translate_entoet(example: _OblExample) -> _OblExample:
    en_lbl = example["label"]
    example["label"] = _EN_TO_ET[en_lbl]
    return example


def load_obl(
    tokenizer: PreTrainedTokenizerFast,
    arch: Architecture,
    *,
    n_shot: int = 0,
    n_virtual: int = 0,
    split: Split | None = None,
) -> tuple[DatasetDict, DatasetInfo]:
    """
    Load, tokenize, and prompt-format the OBL dataset.

    OBL is always loaded in full from a fixed CSV, so `split` has no effect;
    it exists only so this function matches the shared `DatasetLoader` shape.
    """
    logger.debug("load obl csv from github permalink")
    raw = Dataset.from_csv(_PERMALINK, sep=";")

    logger.debug("rename 'type' to 'label'")
    raw = raw.rename_column("type", "label")

    logger.debug("translate labels")
    raw = raw.map(_translate_entoet)

    s1 = raw.train_test_split(test_size=1000, seed=0)
    s2 = s1["train"].train_test_split(test_size=128, seed=0)
    split = cast(
        "Split",
        {
            "train": s2["train"],
            "validation": s2["test"],
            "test": s1["test"],
        },
    )

    data = DatasetDict(cast("dict", split))

    max_shots = len(s2["train"])
    if n_shot > max_shots:
        msg = f"requested more than {max_shots} examples"
        raise ValueError(msg)
    elif n_shot > 0:
        sampled = s2["train"].select(range(n_shot))
        shots = [_format_shot(s) for s in sampled]
    else:
        shots = []

    logger.debug("tokenize obl")
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
        system_prompt=obl_sys_prompt(tokenizer, arch),
    )

    return data, info
