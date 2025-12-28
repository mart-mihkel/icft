TB_LOG_DIR = out
NOTEBOOK_DIR = notebooks
DOCUMENT_VIEWER = zathura

install:
	uv sync
	uv pip install --editable .

test:
	@uv run pytest

types:
	@uv run ty check

lint:
	@uv run ruff check --fix

format:
	@uv run ruff format

check: format lint types test

notebooks:
	uv run marimo edit $(NOTEBOOK_DIR)

typesetting:
	typst watch typesetting/main.typ --open $(DOCUMENT_VIEWER)

tensorboard:
	uv run tensorboard --logdir $(TB_LOG_DIR)
