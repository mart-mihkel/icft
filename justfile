set dotenv-load

[doc('list recipes')]
[group('meta')]
[private]
@info:
    just --list

[arg('workers', long, short, pattern='\d+', help='max flash-attn compile jobs')]
[arg('backend', long, short, pattern='cpu|cu132', help='pytorch backend')]
[doc('create or sync virtualenv')]
[group('build')]
sync backend='cpu' workers='4':
    MAX_JOBS={{ workers }} uv sync \
        --compile-bytecode \
        --extra notebooks \
        --extra {{ backend }}

[arg('host', long, short, help='remote host and target dir')]
[arg('commit', long, short, pattern='true|false', value='true', help='no dry run')]
[doc('upload working tree to remote host')]
[group('build')]
rsync commit='false' host=env('REMOTE'):
    rsync --verbose --archive --delete \
        {{ if commit == 'true' { '' } else { '--dry-run' } }} \
        --exclude-from .gitignore \
        --exclude .pytest_cache \
        --exclude .ruff_cache \
        --exclude .git \
        . {{ host }}

[arg('dry', long, short, pattern='true|false', value='true', help='dry run')]
[doc('run pre-commit checks')]
[group('lint')]
check dry='false':
    uv run --no-sync ruff format {{ if dry == 'true' { '--check' } else { '' } }}
    uv run --no-sync ruff check {{ if dry == 'true' { '' } else { '--fix' } }}
    uv run --no-sync ty check {{ if dry == 'true' { '' } else { '--fix' } }}

[arg('workers', long, short, pattern='\d+', help='number of test runners')]
[arg('slow', long, short, pattern='true|false', value='true', help='run slow tests')]
[doc('run test suite')]
[group('lint')]
test workers='2' slow='false':
    uv run --no-sync pytest \
        {{ if slow == 'true' { '--run-slow' } else { '' } }} \
        --numprocesses {{ workers }} \
        --quiet 
