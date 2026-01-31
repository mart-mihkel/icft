TB_LOG_DIR = out
NOTEBOOK_DIR = notebooks
DOCUMENT_VIEWER = zathura

install:
	uv sync
	uv pip install --editable .

pre-commit:
	uv run ruff check --fix
	uv run ruff format
	uv run ty check

marimo:
	uv run marimo edit $(NOTEBOOK_DIR)

tensorboard:
	uv run tensorboard --logdir $(TB_LOG_DIR)

typst:
	typst watch typesetting/main.typ --open $(DOCUMENT_VIEWER)
