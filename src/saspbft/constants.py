"""Global constants."""

from pathlib import Path

IGNORE_TOKEN = -100
LOGDIR = Path("log")
PAD_MULTIPLE = 8

# tokenizers without a real length limit report a ~1e30 sentinel instead; treat
# anything at or above this threshold as "no limit configured"
UNSET_MAX_LENGTH = int(1e15)
