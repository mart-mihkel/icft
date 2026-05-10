from textwrap import dedent
from typing import Literal, TypedDict, cast

from datasets.arrow_dataset import Dataset
from datasets.dataset_dict import DatasetDict
from transformers import BatchEncoding, PreTrainedTokenizerFast

from instruct.logging import logger
from instruct.types import Architecture, DatasetInfo

_permalink = (
    "https://raw.githubusercontent.com/estnltk/estnltk-model-data/"
    "c6f373a49d29a5a4b5e92326e5309b173bfcd3d1"
    "/phrase_removal/benchmarks/obl_phrases/obl_max/data/obl_all_hand_annotated.csv"
)

type OblLabel = Literal["seotud", "vaba", "ebaloomulik", "liigne koma", "kaheldav"]


class OblExample(TypedDict):
    id: int
    fpath: str
    sentence: str
    remove_start: int
    remove_end: int
    removed: str
    label: OblLabel
    short_sent: str
    cons_score: float
    ual: float
    la: float


entoet: dict[str, OblLabel] = {
    "bound": "seotud",
    "free": "vaba",
    "unnatural": "ebaloomulik",
    "redundant comma": "liigne koma",
    "dubious": "kaheldav",
}

id2label: dict[int, OblLabel] = {
    0: "seotud",
    1: "ebaloomulik",
    2: "liigne koma",
    3: "vaba",
    4: "kaheldav",
}

label2id: dict[OblLabel, int] = {
    "seotud": 0,
    "ebaloomulik": 1,
    "liigne koma": 2,
    "vaba": 3,
    "kaheldav": 4,
}

shots = [
    (
        "Lause: Ilma vihmavarjuta ei oleks me kuiva nahaga koju jõudnud .\n"
        "Fraas: Ilma vihmavarjuta\n"
        "Kategooria: seotud\n"
    ),
    (
        "Lause: Ta lõpetas töö hilisõhtul kontoris ja läks siis koju .\n"
        "Fraas: kontoris\n"
        "Kategooria: vaba\n"
    ),
    (
        "Lause: Kiiresti joostes trepist üles ta jõudis lõpuks kohale .\n"
        "Fraas: Kiiresti joostes trepist üles\n"
        "Kategooria: ebaloomulik\n"
    ),
    (
        "Lause: Mõni aeg tagasi , kohtusime vana sõbraga juhuslikult tänaval .\n"
        "Fraas: Mõni aeg tagasi\n"
        "Kategooria: üleliigne koma\n"
    ),
    (
        "Lause: Ta rääkis suure innuga oma uuest projektist töö juures .\n"
        "Fraas: töö juures\n"
        "Kategooria: kaheldav\n"
    ),
]


def _enc_sys_prompt(sep: str) -> str:
    return dedent(f"""
        Mis on seos fragmendi ja lause vahel,
        kas see on seotud, vaba, kaheldav, ebaloomulik või liigne koma?{sep}
    """).strip()


def _enc_prompt(example: OblExample, sep: str) -> str:
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


def _dec_prompt(example: OblExample) -> str:
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


def _encdec_prompt(example: OblExample) -> str:
    return dedent(f"""
        lause: {example["sentence"]}
        fraas: {example["removed"]}
        kategooria:
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
    example: OblExample,
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
    example: OblExample,
    tokenizer: PreTrainedTokenizerFast,
    arch: Architecture,
    n_shot: int,
) -> BatchEncoding:
    sys = _get_sys_prompt(tokenizer, arch)
    prompt = _get_prompt(tokenizer, arch, example, n_shot)
    label = example["label"]

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
        prompt_enc["label"] = label2id[label]
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


def _translate_entoet(example: OblExample) -> OblExample:
    en_lbl = example["label"]
    example["label"] = entoet[en_lbl]
    return example


def load_obl(
    tokenizer: PreTrainedTokenizerFast,
    arch: Architecture,
    n_shot: int = 0,
) -> tuple[DatasetDict, DatasetInfo]:
    logger.debug("load obl csv from github permalink")
    raw = Dataset.from_csv(_permalink, sep=";")

    logger.debug("rename 'type' to 'label'")
    raw = raw.rename_column("type", "label")

    logger.debug("translate labels")
    raw = raw.map(_translate_entoet)

    s1 = raw.train_test_split(test_size=1000, seed=0)
    s2 = s1["train"].train_test_split(test_size=128, seed=0)
    split = {"train": s2["train"], "dev": s2["test"], "test": s1["test"]}
    data = DatasetDict(cast(dict, split))

    logger.debug("tokenize obl")
    cols = [
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
