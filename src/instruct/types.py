"""Shared type aliases and typed dicts for datasets and models."""

from typing import Literal, TypedDict

type Architecture = Literal["encoder", "decoder", "encoder-decoder"]
type DatasetName = Literal["multinerd", "estner", "boolq", "wic", "obl"]
type PrefixInit = Literal["pretrained", "random"]


class DatasetInfo(TypedDict):
    """Label mappings and system prompt needed to load and prompt a dataset."""

    id2label: dict[int, str]
    label2id: dict[str, int]
    system_prompt: str
