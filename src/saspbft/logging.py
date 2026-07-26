"""Rich-backed logging setup."""

import logging

import click
import rich.traceback
from rich.logging import RichHandler
from rich.traceback import install as install_rich_traceback

install_rich_traceback(suppress=[rich, click])

logging.basicConfig(
    format="%(message)s",
    handlers=[RichHandler(show_path=False, show_time=False)],
)

logger = logging.getLogger("saspbft")
"""Global logger."""
