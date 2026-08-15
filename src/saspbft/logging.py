"""Rich-backed logging setup."""

import logging
import sys
from functools import partialmethod
from typing import TYPE_CHECKING, TextIO, cast

import click
import rich.traceback
import tqdm.std
from click import (
    Context,
    HelpFormatter,
    style,
)
from rich.console import Console
from rich.logging import RichHandler

if TYPE_CHECKING:
    from collections import Counter
    from collections.abc import Iterable

logger = logging.getLogger("saspbft")
"""Global logger."""


class _ColorHelpFormatter(HelpFormatter):
    """Help formatter that colors headings, usage, and option/command names."""

    def write_usage(self, prog: str, args: str = "", prefix: str | None = None) -> None:
        prefix = prefix if prefix is not None else "Usage: "
        colored_prefix = style(prefix, fg="green", bold=True)
        super().write_usage(prog, args, prefix=colored_prefix)

    def write_heading(self, heading: str) -> None:
        super().write_heading(style(heading, fg="yellow", bold=True))

    def write_dl(
        self,
        rows: Iterable[tuple[str, str]],
        col_max: int = 30,
        col_spacing: int = 2,
    ) -> None:
        super().write_dl(
            [(style(name, fg="cyan"), description) for name, description in rows],
            col_max=col_max,
            col_spacing=col_spacing,
        )


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


Context.formatter_class = _ColorHelpFormatter
