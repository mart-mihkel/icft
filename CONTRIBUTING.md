# Contributing

## Setup

Use [uv](https://docs.astral.sh/uv/) for package management.

Setup a virtualenv with the torch backend for cpu or cuda. When using cuda you
should also have cuda-toolkit on the system to compile flash attention.

```bash
make install BACKEND=[cpu|cu132] MAX_JOBS=[n-jobs]
```

You can limit the number of compile workers by setting the `MAX_JOBS` variable.

### Testing

All regular tests can be run with:

```bash
pytest
```

Run a single test file or case instead of the full suite where possible:

```bash
pytest test/test_metrics.py::test_filter_gibberish
```

Run parameterized tests by matching a python expression

```bash
pytest -k gemma test/test_models.py
```

Slow tests are skipped by default. CI runs them with `--run-slow`:

```bash
pytest --run-slow
```

## Pre-Commit

Pre-commit checks require [shellcheck](https://www.shellcheck.net/) to be
installed.

Run all pre-commit checks with:

```bash
make check
```

To run the individual tools directly:

```bash
ruff format --check
ruff check
ty check
```

## Notebooks

Notebooks are python files in the [notebooks](./notebooks) directory, built with
`marimo` and can be edited or run trough the command line

```bash
marimo edit notebooks
```
