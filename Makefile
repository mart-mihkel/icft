.PHONY: setup
setup:
	uv sync

.PHONY: marimo
marimo: setup
	uv run marimo edit

.PHONY: format
format: setup
	uv run ruff format

.PHONY: lint
lint: setup
	uv run ruff check

.PHONY: types
types: setup
	uv run ty check

.PHONY: check
check: setup format lint types

.PHONY: watch
watch:
	typst watch typesetting/main.typ --open zathura
