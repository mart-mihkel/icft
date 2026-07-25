# Study About Soft-Prompt Based Fine-Tuning

See [CONTRIBUTING.md](./CONTRIBUTING.md) for setup and development instructions.

## Usage

The `cli` installed in the virtualenv contains scripts for fine-tuning,
prompt-tuning, few-shot learning, mlfow utilities and submittin SLURM jobs.

```bash
cli --help
```

Example SLURM jobs are defined in [slurm.py](./src/saspbft/slurm.py)

## Tracking

Experiment are tracked to `mlflow` and can be seen by serving the ui

```bash
mlflow ui
```
