# Contributing to OMEGA

Thank you for your interest in contributing to OMEGA!

## Development Setup

```bash
# Clone
git clone https://github.com/omega-memory/core.git
cd core

# Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install with dev dependencies
pip install -e ".[dev]"

# Download the embedding model
omega setup

# Verify everything works
omega doctor
```

Requires Python 3.11+.

## Running Tests

```bash
pytest tests/                    # Run all tests
pytest tests/ -x                 # Stop on first failure
pytest tests/test_cli.py         # Run a specific test file
pytest tests/ -k "test_query"    # Run tests matching a pattern
pytest tests/ --tb=short         # Shorter tracebacks
```

Tests use `pytest-asyncio` for async handler tests. The `asyncio_mode = "auto"` setting in `pyproject.toml` means you don't need to decorate async tests manually.

Some tests are marked `@pytest.mark.slow` and are excluded by default. Run them with:

```bash
pytest tests/ -m slow
```

## Code Style

We use [ruff](https://docs.astral.sh/ruff/) for linting and formatting:

```bash
ruff check src/             # Lint
ruff check src/ --fix       # Auto-fix lint issues
ruff format src/            # Format code
```

Configuration is in `pyproject.toml` under `[tool.ruff]`. Key settings:
- Target: Python 3.11
- Line length: 120 characters
- A few rules are intentionally ignored (see `[tool.ruff.lint]`)

## Testing the MCP Server Locally

You can test the MCP server without Claude Code using any MCP client, or by sending JSON-RPC requests directly:

```bash
# Start the server in stdio mode
python -m omega.server.mcp_server

# Or use the mcp CLI inspector (if installed)
npx @modelcontextprotocol/inspector python -m omega.server.mcp_server
```

The server communicates via JSON-RPC over stdio. Tool schemas are defined in `src/omega/server/tool_schemas.py` and handlers in `src/omega/server/handlers.py`.

## Project Structure

```
src/omega/
  __init__.py          # Package exports
  bridge.py            # High-level API (used by handlers)
  sqlite_store.py      # SQLite backend
  graphs.py            # Embedding model + semantic search
  cli.py               # CLI commands (omega query, setup, etc.)
  server/
    mcp_server.py      # MCP server entry point
    tool_schemas.py     # Tool definitions (26 tools)
    handlers.py         # Tool handler implementations
```

## Pull Request Process

1. **Branch from `main`** — use descriptive names like `fix/model-download` or `feat/json-output`
2. **Keep PRs focused** — one feature or fix per PR
3. **All tests must pass** — `pytest tests/` with no failures
4. **Lint clean** — `ruff check src/` with no errors
5. **Write a clear description** — explain what changed and why

### Commit Style

We follow conventional commits:

- `feat:` — new feature
- `fix:` — bug fix
- `docs:` — documentation only
- `chore:` — maintenance (version bumps, CI, etc.)
- `refactor:` — code change that neither fixes a bug nor adds a feature

## What to Contribute

- Bug fixes (check [open issues](https://github.com/omega-memory/core/issues))
- Documentation improvements
- Test coverage for uncovered code paths
- Performance optimizations
- New memory tool ideas (open an issue first to discuss)

## Developer Certificate of Origin

By contributing, you certify that your contribution is your own work and you have the right to submit it under the Apache-2.0 license. We use the [Developer Certificate of Origin](https://developercertificate.org/) (DCO).

Sign your commits with `git commit -s` to add the DCO sign-off.

## Questions?

Open a [GitHub Discussion](https://github.com/omega-memory/core/discussions) or file an [issue](https://github.com/omega-memory/core/issues).
