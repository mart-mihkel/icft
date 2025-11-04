.PHONY: setup
setup:
	uv sync

.PHONY: marimo
marimo:
	uv run marimo edit

.PHONY: format
format:
	uv run ruff format

.PHONY: lint
lint:
	uv run ruff check

.PHONY: watch
watch:
	typst watch thesis/main.typ --open zathura
