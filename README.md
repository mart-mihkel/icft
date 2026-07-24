# ML Based LM Instructing

## Development

Use [uv](https://docs.astral.sh/uv/) for package management.

Setup a virtualenv with torch backend for cpu or cuda. When using cuda you
should also have cuda-toolkit on the system to compile flash attention.

```bash
make install BACKEND=[cpu|cu132]
```

You can limit the number of compile workers by setting the `MAX_JOBS` variable.

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

## Notebooks

Notebooks are python files in the [notebooks](./notebooks) directory, built
with `marimo` and can be edited or run trough the command line

```bash
marimo edit notebooks
```
