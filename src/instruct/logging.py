"""Rich-backed console logging setup, suppressing noisy third-party tracebacks."""

import logging
import sqlite3

import accelerate
import datasets
import evaluate
import httpx
import numpy
import peft
import polars
import torch
import transformers
from rich.console import Console
from rich.logging import RichHandler
from rich.traceback import install

_suppress = [
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
]

_console = Console(width=80)
_handler = RichHandler(
    show_path=False,
    show_time=False,
    console=_console,
    rich_tracebacks=True,
    tracebacks_suppress=_suppress,
)

install(console=_console, suppress=_suppress)
logging.basicConfig(format="%(message)s", handlers=[_handler])
logger = logging.getLogger("instruct")
