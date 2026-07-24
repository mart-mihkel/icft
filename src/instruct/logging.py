"""Rich-backed logging setup."""

import contextlib
import logging
import sqlite3
from typing import TYPE_CHECKING

import accelerate
import datasets
import evaluate
import httpx
import numpy
import peft
import polars
import torch
import transformers
from rich.logging import RichHandler

if TYPE_CHECKING:
    from types import ModuleType


_dev_suppress = []
with contextlib.suppress(ImportError):
    import pytest

    _dev_suppress: list[ModuleType] = [pytest]

_suppress: list[ModuleType] = [
    transformers,
    accelerate,
    datasets,
    evaluate,
    sqlite3,
    polars,
    torch,
    httpx,
    numpy,
    peft,
    *_dev_suppress,
]

logging.basicConfig(
    format="%(message)s",
    handlers=[
        RichHandler(
            show_path=False,
            show_time=False,
            rich_tracebacks=True,
            tracebacks_suppress=_suppress,
        )
    ],
)

logger = logging.getLogger("instruct")
"""Global logger."""
