REMOTE ?=
MAX_JOBS=4
BACKEND=cpu

all: install check test

install:
	@MAX_JOBS=$(MAX_JOBS) uv sync \
		--compile-bytecode \
		--extra notebooks \
		--extra $(BACKEND)

check:
	@uv run --no-sync ruff format
	@uv run --no-sync ruff check --fix
	@uv run --no-sync ty check

test:
	@uv run --no-sync pytest --quiet --numprocesses auto

push:
	@rsync --verbose --archive --delete \
		--exclude-from .gitignore \
		--exclude .pytest_cache \
		--exclude .ruff_cache \
		--exclude .git \
		. $(REMOTE)

pull:
	@rsync --verbose --archive \
		$(REMOTE)/mlflow.db \
		$(REMOTE)/log \
		.
