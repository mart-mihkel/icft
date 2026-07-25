"""Rich-backed logging setup."""

import logging

from rich.logging import RichHandler

logging.basicConfig(
    format="%(message)s",
    handlers=[RichHandler(show_path=False, show_time=False)],
)

logger = logging.getLogger("saspbft")
"""Global logger."""
