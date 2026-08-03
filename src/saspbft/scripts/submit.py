"""Submit predefined training jobs to Slurm."""

import re
import shutil
import subprocess
import sys

from rich.console import Console

from saspbft.constants import SLURMDIR
from saspbft.logging import logger
from saspbft.slurm import JOBS, command, sbatch_args


def show() -> None:
    """Show all predefined jobs."""
    names = [j.job_name for j in JOBS]
    console = Console(stderr=True)
    console.print("[bold]Available jobs[/bold]:")
    for name in sorted(set(names)):
        console.print(f"  [cyan]-[/cyan] {name}")


def submit(job_name: str) -> None:
    """Submit one SLURM job per model."""
    jobs = [j for j in JOBS if re.fullmatch(job_name, j.job_name) is not None]

    if len(jobs) == 0:
        logger.error("no matches for job: '%s'", job_name)
        show()
        sys.exit(1)

    logger.info("submitting %d jobs to SLURM", len(jobs))

    if shutil.which("sbatch") is None:
        logger.error("'sbatch' not found in PATH, are you on a Slurm login node?")
        sys.exit(1)

    SLURMDIR.mkdir(parents=True, exist_ok=True)

    for job in jobs:
        for model in job.models:
            wrap = command(model, job)
            cmd = ["sbatch", *sbatch_args(model, job), f"--wrap={wrap}"]

            logger.info(wrap)
            subprocess.run(cmd, check=True)  # noqa: S603
