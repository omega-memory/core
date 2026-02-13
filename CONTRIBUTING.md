# Contributing to OMEGA

Thank you for your interest in contributing to OMEGA! This document provides guidelines for contributing to the project.

## Getting Started

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/YOUR-USERNAME/core.git
   cd core
   ```
3. Install in development mode:
   ```bash
   pip install -e ".[dev]"
   ```
4. Run the tests:
   ```bash
   pytest tests/ -x -q
   ```

## Development Workflow

1. Create a branch for your change:
   ```bash
   git checkout -b your-feature-name
   ```
2. Make your changes
3. Run tests and linting:
   ```bash
   pytest tests/ -x -q
   ruff check src/
   ```
4. Commit with a clear message describing the change
5. Push and open a pull request

## Code Style

- Python 3.11+ required
- We use [ruff](https://github.com/astral-sh/ruff) for linting
- Line length: 120 characters
- Follow existing patterns in the codebase

## Testing

- All new features should include tests
- All bug fixes should include a regression test
- Tests live in `tests/` and use pytest + pytest-asyncio
- Run the full suite: `pytest tests/ -x -q`

## Developer Certificate of Origin

By contributing to this project, you certify that your contribution was created in whole or in part by you and that you have the right to submit it under the Apache-2.0 license. This is the [Developer Certificate of Origin (DCO)](https://developercertificate.org/).

## Reporting Issues

- Use [GitHub Issues](https://github.com/omega-memory/core/issues) for bug reports and feature requests
- For security vulnerabilities, see [SECURITY.md](SECURITY.md)

## License

By contributing, you agree that your contributions will be licensed under the Apache License 2.0.
