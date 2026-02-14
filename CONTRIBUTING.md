# Contributing to OMEGA

Thank you for your interest in contributing to OMEGA!

## Development Setup

### Prerequisites

- Python 3.11 or later
- Git

### Clone and Install

```bash
# 1. Fork the repository on GitHub, then clone your fork
git clone https://github.com/YOUR-USERNAME/core.git
cd core

# 2. Create a virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install in editable mode with dev dependencies
pip install -e ".[dev]"

# 4. Download the embedding model
omega setup

# 5. Verify everything works
omega doctor
```

### Running Tests

```bash
# Run the full test suite
pytest tests/

# Run with verbose output
pytest tests/ -v

# Run a specific test file
pytest tests/test_sqlite_store.py

# Run with coverage report
pytest tests/ --cov=omega --cov-report=term-missing

# Skip slow tests (default behavior, configured in pyproject.toml)
pytest tests/ -m "not slow"

# Include slow tests
pytest tests/ -m ""
```

Note: `asyncio_mode = "auto"` is configured in `pyproject.toml`, so async tests
run without extra markers.

## Code Style

We use [ruff](https://docs.astral.sh/ruff/) for both linting and formatting.
Configuration lives in `pyproject.toml` under `[tool.ruff]`.

```bash
# Check for lint issues
ruff check src/

# Auto-fix lint issues where possible
ruff check src/ --fix

# Format code
ruff format src/

# Check formatting without modifying files
ruff format src/ --check
```

Key settings:
- Target: Python 3.11 (`target-version = "py311"`)
- Line length: 120 characters
- Lazy imports are allowed (`E402` is ignored by design)

## Testing the MCP Server Locally

You can test the MCP server without a full Claude Code setup using `omega` CLI
commands directly:

```bash
# Store a memory
omega remember "This is a test memory about Python development"

# Query memories
omega query "Python development"

# Check server health and config
omega doctor
```

For direct MCP protocol testing, you can pipe JSON-RPC messages to the server:

```bash
# Start the MCP server in stdio mode
omega serve
```

## Making Changes

1. **Create a branch** from `main`:
   ```bash
   git checkout -b fix/issue-number-short-description
   # or: feat/issue-number-short-description
   # or: docs/issue-number-short-description
   ```

2. **Make your changes.** Keep commits small and focused.

3. **Run checks** before committing:
   ```bash
   ruff check src/
   ruff format src/
   pytest tests/
   ```

4. **Commit** with a descriptive message and sign off:
   ```bash
   git commit -s -m "fix: short description of what you fixed (#issue-number)"
   ```

5. **Push** your branch and open a Pull Request against `main`.

### Branch Naming

| Prefix   | Use for                  |
|----------|--------------------------|
| `fix/`   | Bug fixes                |
| `feat/`  | New features             |
| `docs/`  | Documentation changes    |
| `test/`  | Test improvements        |
| `perf/`  | Performance improvements |

### Commit Style

Use [Conventional Commits](https://www.conventionalcommits.org/):
- `fix: correct error message for missing model`
- `feat: add batch embedding support`
- `docs: expand contributing guide`

## Pull Request Process

1. **Title**: Use the same conventional commit format as your commit message
2. **Description**: Explain what changed and why. Reference the issue number
3. **Tests**: Add or update tests for any behavioral changes
4. **Lint**: Ensure `ruff check` and `ruff format --check` pass
5. **DCO**: Sign your commits with `-s` flag (see below)

Reviewers look for:
- Tests covering the change
- Clean ruff output
- No regressions in existing tests
- Clear commit messages

## What to Contribute

- Bug fixes
- Documentation improvements
- Test coverage
- Performance optimizations
- New memory tool ideas (open an issue first to discuss)

## Developer Certificate of Origin

By contributing, you certify that your contribution is your own work and you have the right to submit it under the Apache-2.0 license. We use the [Developer Certificate of Origin](https://developercertificate.org/) (DCO).

Sign your commits with `git commit -s` to add the DCO sign-off.

## Questions?

Open a [GitHub Discussion](https://github.com/omega-memory/core/discussions) or file an [issue](https://github.com/omega-memory/core/issues).
