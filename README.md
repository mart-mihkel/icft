# Study About Soft-Prompt Based Fine-Tuning

See [CONTRIBUTING.md](./CONTRIBUTING.md) for setup and development instructions.

## Usage

The `cli` installed in the virtualenv contains scripts for fine-tuning,
prompt-tuning, few-shot learning and utilities.

```bash
cli --help
```

Example experiments are in the [run](./run) directory.

## Tracking

Experiment are tracked to `mlflow` and can be seen by serving the ui

```bash
mlflow ui
```
