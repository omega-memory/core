# Changelog

All notable changes to OMEGA (`omega-memory`) will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.10.8] - 2026-02-24

### Fixed

- Test fixture: `CoordinationManager` in conftest.py now passes `cloud_sync=False` to prevent cloud sync attempts during tests.
- `__init__.py` version string now matches `pyproject.toml` (was stuck at 0.10.6).

## [0.10.7] - 2026-02-24

### Added

- MCP server now sends instructions during initialization telling Claude to call `omega_welcome()` and use memory tools proactively. Removes need for manual project instruction configuration in Claude Desktop.
- Synced core files from private repo: `hooks-core.json`, `task_utils.py`, `test_improvements.py`, `test_sqlite_store.py`.
- Pro-only test guards (`_skip_pro`) added to `test_sqlite_store.py` for features that depend on pro-only infrastructure.

## [0.10.6] - 2026-02-20

### Added

- WAL checkpoint management: TRUNCATE on init, PASSIVE on close, and periodic PASSIVE checkpoint every N writes (configurable via `OMEGA_WAL_CHECKPOINT_INTERVAL` env var). Prevents unbounded WAL growth under multi-process contention.
- Auto-backup on startup: creates a JSON backup when the most recent is >24h old, keeps max 5 rotated in `~/.omega/backups/`.
- `_run_sql()` retry wrapper for individual SQL statements under lock contention.
- Thread-safe query cache with `_cache_lock` protecting all reads/writes.
- Smart partial cache invalidation: only evicts cache entries with trigram overlap >= 0.20 with new content, instead of full wipe.

### Fixed

- `find_similar()` now filters expired and superseded memories, with 2x over-fetch to maintain result count.

## [0.10.5] - 2026-02-20

### Added

- Windows/WSL installation guide in README with step-by-step setup and 5 WSL-specific gotchas.
- `src/omega/db_utils.py` shared SQLite retry helpers (`retry_on_locked`, `retry_write_on_locked`) for database-locked error recovery.

### Fixed

- SQLite connections in CLI commands now use `timeout=30` to prevent WAL lock hangs under multi-process contention. Doctor health checks use `timeout=5`.
- `SESSION_START` surfacing thresholds broadened (0.60/0.45/0.15 to 0.45/0.40/0.10) for better context at session startup.
- Three noise filters added to auto-capture: infrastructure events, zero-token outcomes, and JSON blob decisions.
- Cross-session dedup exception for decisions (same decision restated across sessions now correctly deduplicates).
- Shortened capture output messages ("Deduped", "Evolved") for cleaner hook output.
- Lowered dedup thresholds: session summaries 0.95 to 0.75, decisions 0.85 to 0.80.
- `exc_info=True` added to all `logger.error()` calls in bridge.py.

## [0.10.4] - 2026-02-20

### Added

- Full `--json` test coverage for `query`, `status`, `timeline`, `stats`, and `activity` CLI commands (closes #9).

### Changed

- Session summary TTL changed from SHORT_TERM to EPHEMERAL (1h) to prevent accumulation.
- Generic `memory` type TTL changed from LONG_TERM to SHORT_TERM (1d) for faster expiry.
- `consolidate` prune_days default lowered from 30 to 14 days.
- Added `exc_info=True` to all 30 `logger.error()` calls in handlers for better tracebacks.
- Eliminated all `datetime.utcnow()` deprecation warnings (Python 3.12+).

## [0.10.3] - 2026-02-20

### Changed

- Removed noisy stderr warning when ONNX model is not downloaded (logger.warning still fires for debugging).
- Version alignment: `__init__.py` and `pyproject.toml` now both report 0.10.3.

## [0.10.0] - 2026-02-16

### Changed

- **MCP is now an optional dependency.** `pip install omega-memory` installs the core library only (storage, retrieval, embeddings). For MCP server integration with Claude Code, Cursor, Windsurf, or Zed, install with `pip install omega-memory[server]`. This reduces the base install size and eliminates the MCP server process for users who only need the Python API.
- **Expanded public Python API.** 9 new functions exported from the top-level `omega` package: `batch_store`, `record_feedback`, `deduplicate`, `get_session_context`, `get_activity_summary`, `create_reminder`, `list_reminders`, `dismiss_reminder`, `get_due_reminders`.

### Added

- **`omega setup --hooks-only`** configures Claude Code hooks and CLAUDE.md without registering the MCP server, saving ~600MB RAM per session. Hooks call bridge.py directly.
- **Direct Python API** for scripts, CI/CD, and automation without running an MCP server:
  ```python
  from omega import store, query, remember
  store("Always use TypeScript strict mode", "user_preference")
  results = query("TypeScript preferences")
  ```
- MCP import guard in `mcp_server.py` prints a clear error message with install instructions if the `mcp` package is missing.

### Migration

If you use OMEGA with an MCP client (Claude Code, Cursor, etc.), update your install command:

```bash
# Before (0.9.x)
pip install omega-memory

# After (0.10.0+)
pip install omega-memory[server]
```

If you only use OMEGA as a Python library, `pip install omega-memory` continues to work and is now lighter.

## [0.9.0] - 2026-02-15

### Added

- **Contradiction detection** — ingest-side intelligence that auto-detects conflicting memories on store. When a new decision contradicts an existing one, the older memory is automatically superseded with a `contradicts` relationship edge, keeping the knowledge graph consistent without manual cleanup.
- **Atomic fact splitting** — compound memories (e.g. "Project uses React and deploys to Vercel") are automatically decomposed into individual fact nodes during ingestion, improving retrieval precision for targeted queries.
- **Corpus hygiene** — automated deduplication of near-duplicate memories, reducing noise and token waste in search results over long-lived sessions.
- **Compact MCP tool responses** — all MCP tool responses optimized for token efficiency, reducing context window consumption when agents interact with OMEGA.

## [0.8.0] - 2026-02-14

### Added

- **`omega status --json`** — machine-readable JSON output for scripted access to memory count, DB size, model status, and vector search availability.
- **`omega export`** — export memories to a JSON file, with optional `--type` filter (e.g. `omega export --type decision decisions.json`).
- **`omega import`** — import memories from a JSON file, with optional `--clear` to replace existing data.

### Changed

- **CONTRIBUTING.md** — expanded with full dev setup, test commands, code style guide, MCP server testing instructions, project structure, and PR process.

## [0.7.3] - 2026-02-14

### Fixed

- **Welcome briefing false positive** — model-missing warning in `omega_welcome` now checks for model files on disk instead of backend activation state, which is lazy-loaded and always None at welcome time.

## [0.7.2] - 2026-02-14

### Fixed

- **Missing model warning** — when the ONNX embedding model is not downloaded, OMEGA now shows a clear actionable error ("Run 'omega setup' to download the model") instead of silently falling back to degraded text-only search. Warnings appear in CLI queries, model loading, and the MCP welcome briefing.

## [0.7.1] - 2026-02-14

### Fixed

- **MiniLM model download** — tokenizer and config files were fetched from the wrong HuggingFace path (`onnx/` subdirectory instead of repo root), causing 404 errors during `omega setup` for first-time users without the bge model.

## [0.7.0] - 2026-02-14

### Added

- **Multi-client setup** — `omega setup --client cursor|windsurf|zed` writes MCP config to each editor's config file. Claude Code remains the default with full hooks and instruction injection.
  - Cursor: `~/.cursor/mcp.json`
  - Windsurf: `~/.codeium/windsurf/mcp_config.json`
  - Zed: `~/.config/zed/settings.json`

### Changed

- Setup auto-detect now lists all supported clients when Claude Code is not found.

## [0.6.1] - 2026-02-14

### Added

- **Entity auto-capture** — `resolve_project_entity()` wired into `bridge.auto_capture()` for automatic entity scoping.
- **Smithery.yaml** — configuration file for Smithery.ai directory listing.
- **Demo GIF** — animated terminal demo in README showing cross-session memory recall.

### Changed

- README restructured: leads with problem statement, demo GIF, and examples section.

### Fixed

- **SQLite lock contention** — increased `busy_timeout` from 5s to 30s and added retry-with-backoff on all write paths. Fixes "database is locked" errors when multiple Claude Code sessions share the same `omega.db`.

## [0.6.0] - 2026-02-13

### Added

- **Forgetting Audit Trail** — every deletion logged with reason (TTL, LRU, consolidation, feedback, user).
- **Decay Curves** — old unaccessed memories rank lower in search results. Preferences and errors exempt. Floor at 0.35.
- **Conflict Detection** — contradictions auto-detected on store. Decisions auto-resolve (newest wins), lessons get flagged for manual review.
- 31 new tests for forgetting intelligence features.

## [0.5.0] - 2026-02-13

### Initial Open Source Release

OMEGA — persistent memory for AI coding agents. First public release under Apache-2.0.

#### Core Memory System
- **25 MCP tools** for storing, querying, and managing long-term memory
- Semantic search via bge-small-en-v1.5 embeddings + sqlite-vec
- FTS5 full-text search for exact phrase queries
- Graph relationships (related, supersedes, contradicts) with BFS traversal
- Memory compaction and consolidation for long-term hygiene
- Timeline views and session-scoped queries
- Context virtualization (checkpoint/resume)

#### Auto-Capture & Surfacing
- Hook system for automatic context capture and memory surfacing
- Session start/stop lifecycle with welcome briefing
- Pre-edit memory surfacing with file-extension-aware re-ranking
- Auto-capture of decisions, lessons, and error patterns

#### Storage & Security
- SQLite + sqlite-vec backend (local-first, no cloud required)
- Optional AES-256-GCM encryption at rest (`pip install omega-memory[encrypt]`)
- Database created with `0o600` permissions
- Export/import for backup and restore

#### Developer Experience
- `omega setup` — one-command installation
- `omega doctor` — health diagnostics
- `omega query/store/remember` — CLI access to memory
- Plugin architecture via entry points for extensibility

[Unreleased]: https://github.com/omega-memory/omega-memory/compare/v0.10.0...HEAD
[0.10.0]: https://github.com/omega-memory/omega-memory/compare/v0.9.0...v0.10.0
[0.9.0]: https://github.com/omega-memory/omega-memory/compare/v0.8.0...v0.9.0
[0.8.0]: https://github.com/omega-memory/omega-memory/compare/v0.7.3...v0.8.0
[0.7.3]: https://github.com/omega-memory/omega-memory/compare/v0.7.2...v0.7.3
[0.7.2]: https://github.com/omega-memory/omega-memory/compare/v0.7.1...v0.7.2
[0.7.1]: https://github.com/omega-memory/omega-memory/compare/v0.7.0...v0.7.1
[0.7.0]: https://github.com/omega-memory/omega-memory/compare/v0.6.1...v0.7.0
[0.6.1]: https://github.com/omega-memory/omega-memory/compare/v0.6.0...v0.6.1
[0.6.0]: https://github.com/omega-memory/omega-memory/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/omega-memory/omega-memory/releases/tag/v0.5.0
