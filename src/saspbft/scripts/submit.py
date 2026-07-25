"""Submit predefined training jobs to Slurm."""

import re
import shutil
import subprocess
import sys

from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel

from saspbft.logging import logger
from saspbft.slurm import JOBS, command

_console = Console(stderr=True)


def submit(job_name: str) -> None:
    """Submit one SLURM job per model."""
    names = [j.job_name for j in JOBS]
    jobs = [j for j in JOBS if re.fullmatch(job_name, j.job_name) is not None]

    if len(jobs) == 0:
        logger.error("no matches for job: '%s'", job_name)

        options = Columns(
            [f"[cyan][bold]{name}[/bold][/cyan]" for name in sorted(set(names))],
            column_first=True,
            equal=True,
        )

        _console.print(
            Panel(
                options,
                title="Available jobs",
                title_align="left",
                border_style="red",
            )
        )

        sys.exit(1)

    logger.info("submitting %d jobs to SLURM", len(jobs))

    if shutil.which("sbatch") is None:
        logger.error("'sbatch' not found in PATH, are you on a Slurm login node?")
        sys.exit(1)

    for job in jobs:
        for model in job.models:
            wrap = command(model, job)
            cmd = [
                "sbatch",
                f"--job-name={job.job_name}",
                f"--time={job.time}",
                f"--mem={job.mem}",
                f"--cpus-per-task={job.cpus}",
                f"--gres={job.gres}",
                "--partition=gpu",
                "--output=log/slurm/%j-%x.out",
                f"--wrap={wrap}",
            ]

            logger.info(wrap)
            subprocess.run(cmd, check=True)  # noqa: S603
