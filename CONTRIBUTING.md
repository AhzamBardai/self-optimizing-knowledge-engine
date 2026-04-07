# Contributing

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install
```

## Development

```bash
make test        # unit tests
make test-int    # integration tests (requires docker-compose up)
make lint        # ruff + mypy
make eval        # RAGAS evaluation on gold dataset
```

## Commit convention

Format: `type(scope): description`

Types: `feat`, `fix`, `test`, `docs`, `chore`, `refactor`, `perf`, `ci`
