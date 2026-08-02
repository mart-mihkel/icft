# Contributing

## Setup

Use [uv](https://docs.astral.sh/uv/) for package management and
[just](https://just.systems/) as the command runner.

Setup a virtualenv with

```bash
just sync
```

By default this installs pytorch for cpu.

When using cuda you should also have cuda-toolkit on the system to compile flash
attention. The `--workers [n]` flag can be used to limit the number of compile
workers.

```bash
just sync --backend cu132 --workers 4
```

### Testing

All regular tests can be run with:

```bash
just test
```

Run a single test file or case instead of the full suite where possible:

```bash
pytest test/test_metrics.py::test_filter_gibberish
```

Run parameterized tests by matching a python expression

```bash
pytest -k gemma test/test_models.py
```

Slow tests are skipped by default. These can be tiggered manually with the
`--slow` flag

```bash
just test --slow
```

## Pre-Commit

Run all pre-commit checks with:

```bash
just check
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
