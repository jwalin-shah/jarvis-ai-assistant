# Contributing to JARVIS

Thank you for your interest in contributing to JARVIS! This guide will help you get started.

## Development Setup

### Prerequisites
- macOS with Apple Silicon (M1/M2/M3)
- Python 3.11+
- Node.js 18+ (for desktop app)
- Full Disk Access granted to Terminal (for iMessage access)

### First-Time Setup

```bash
# Clone and setup
git clone <repository>
cd jarvis-ai-assistant
make setup
```

This will:
1. Create a Python virtual environment
2. Install dependencies via `uv`
3. Download required ML models
4. Initialize the database

### Verify Setup

```bash
make health  # Check project status
make verify  # Run full verification (lint + typecheck + test)
```

## Development Workflow

### Running Tests

**Always use `make test`** - never run pytest directly:

```bash
make test       # Run all tests (output to test_results.txt)
make test-fast  # Stop at first failure
```

After running tests, check `test_results.txt` for results.

### Code Quality

```bash
make lint      # Run ruff linter
make format    # Auto-format code
make typecheck # Run mypy type checker
```

### Before Committing

1. Run `make verify` to ensure code quality
2. Check `test_results.txt` for any failures
3. Run `git diff` to review changes

## Code Style

- **Line length**: 100 characters
- **Python version**: 3.11+ with strict type hints
- **Linting**: ruff (E, F, I, N, W, UP rules)
- **Formatting**: Run `make format` before committing

### Type Hints

All functions should have complete type annotations:

```python
def process_message(text: str, contact_id: str | None = None) -> Message:
    """Process a message and return structured data."""
    ...
```

### Error Handling

All custom errors should inherit from `JarvisError`:

```python
from jarvis.errors import JarvisError

class MyCustomError(JarvisError):
    """Raised when something specific fails."""
    pass
```

### Prompts

All LLM prompts must be defined in `jarvis/prompts.py` - nowhere else.

## Architecture Overview

- **jarvis/**: Core Python library
  - `intent.py`: Intent classification
  - `retrieval.py`: Message retrieval and search
  - `response.py`: Response generation
  - `embeddings.py`: Embedding models
  - `quality/`: Quality assurance system
- **api/**: FastAPI server
- **desktop/**: Tauri + Svelte desktop app
- **tests/**: Unit and integration tests

See `docs/ARCHITECTURE.md` for detailed architecture documentation.

## Pull Request Process

1. Create a feature branch from `main`
2. Make your changes
3. Run `make verify` to ensure quality
4. Write clear commit messages
5. Push and create a PR
6. Ensure CI passes
7. Request review

### Commit Message Format

```
<type>: <short description>

<optional body with more details>

Co-Authored-By: Your Name <email>
```

Types: `feat`, `fix`, `docs`, `refactor`, `test`, `chore`

## Performance Guidelines

JARVIS runs on memory-constrained systems (8GB RAM). Follow these rules strictly:

1. **One Model at a Time**: Never load multiple MLX models simultaneously. Use the shared `MLXModelLoader` singleton via `get_model()`.
2. **Always parallelize**: Use `n_jobs=1` or `n_jobs=2` for scikit-learn operations (memory-constrained)
3. **Always batch**: Process lists together, not one-at-a-time loops
4. **Always cache**: Cache expensive computations (embeddings, model loads)
5. **Stream large data**: Never hold >500MB in RAM; use memmap for large arrays
6. **Binary over JSON**: Use binary encoding for embeddings and large arrays

## Testing Guidelines

- Use descriptive test names: `test_expand_slang_preserves_capitalization`
- Test edge cases: empty inputs, None values, boundary conditions
- Mock expensive operations (model loading, API calls)
- Keep tests fast: use fixtures and caching

## Getting Help

- Check existing documentation in `docs/`
- Open an issue for bugs or feature requests
- Review existing code for patterns and conventions
