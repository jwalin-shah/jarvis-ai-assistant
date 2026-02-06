# Repository Guidelines

## Project Structure
- `jarvis/`: Core Python library (CLI, prompts, retrieval, response generation).
- `api/`: FastAPI server.
- `desktop/`: Tauri + Svelte desktop app (see `desktop/README.md`).
- `core/`, `models/`, `integrations/`, `contracts/`: Shared infrastructure and ML/runtime pieces.
- `benchmarks/`: Evaluation and validation gates.
- `tests/`: Unit and integration tests.
- `scripts/`: Utilities for benchmarks and reporting.
- `docs/`: Design and architecture docs (see `docs/DESIGN.md`).

## Build, Test, and Development Commands
- `make setup`: Install dependencies and configure git hooks.
- `make api-dev`: Run the API server locally on port 8742.
- `make desktop-setup`: Install desktop app dependencies.
- `cd desktop && npm run tauri dev`: Launch the desktop app (API must be running).
- `make test`: Run the full test suite (writes `test_results.txt`).
- `make test-fast`: Stop on first failure.
- `make check`: Run lint + format-check + typecheck.
- `make verify`: Run full verification (checks + tests).
- `make help`: List all available commands.

## Coding Style & Naming Conventions
- Python is formatted with Ruff (`make format`) and linted with Ruff (`make lint`).
- Line length is 100 characters; target Python version is 3.11.
- Lint rules include `E,F,I,N,W,UP` with `E741` ignored; ML scripts have specific per-file ignores.
- Type checking uses strict `mypy` (`make typecheck`).
- All LLM prompts must live in `jarvis/prompts.py`.

## Testing Guidelines
- Tests run with `pytest` via `make test`; coverage includes `jarvis/`, `api/`, `models/`, `core/`, `integrations/`, `contracts/`, `benchmarks/`.
- Test files are named `test_*.py` and test functions use `test_*`.
- Use descriptive names like `test_expand_slang_preserves_capitalization`.

## Commit & Pull Request Guidelines
- Commit format:
  - `<type>: <short description>` with optional body.
  - Types: `feat`, `fix`, `docs`, `refactor`, `test`, `chore`.
- PR flow: branch from `main`, run `make verify`, push, open PR, ensure CI passes, request review.
