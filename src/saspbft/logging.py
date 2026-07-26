"""Rich-backed logging setup."""

import logging
import sys
from functools import partialmethod
from typing import TYPE_CHECKING, TextIO, cast

import click
import rich.traceback
import tqdm.std
from rich.console import Console
from rich.logging import RichHandler

if TYPE_CHECKING:
    from collections import Counter

logger = logging.getLogger("saspbft")
"""Global logger."""


class _TqdmSafeStream:
    """
    Buffer writes and flush per line through `tqdm.write`.

    `tqdm.write` clears and redraws any active progress bars around the write,
    so log lines from other threads/callbacks (e.g. Trainer's eval loop) never
    land on top of a bar's carriage-returned line.
    """

    def __init__(self, stream: TextIO) -> None:
        self._stream = stream
        self._buffer = ""

    def write(self, text: str) -> int:
        self._buffer += text
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            tqdm.std.tqdm.write(line, file=self._stream)
        return len(text)

    def flush(self) -> None:
        self._stream.flush()

    def isatty(self) -> bool:
        return self._stream.isatty()


def setup_logging() -> None:
    """Install the rich log handler, ascii tqdm bars, and rich tracebacks."""
    rich.traceback.install(suppress=[rich, click, tqdm])

    tqdm.std.tqdm.__init__ = partialmethod(tqdm.std.tqdm.__init__, ascii=True)

    logging.basicConfig(
        format="%(message)s",
        handlers=[
            RichHandler(
                console=Console(file=cast("TextIO", _TqdmSafeStream(sys.stderr))),
                show_path=False,
                show_time=False,
            ),
        ],
    )


def format_counts[K](counts: Counter[K]) -> str:
    """Render a Counter as a compact 'key: count' summary, most common first."""
    return ", ".join(f"{key}={count}" for key, count in counts.most_common())
