# Changelog

All notable changes to OMEGA will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

### Fixed

- **SQLite lock contention** — increased `busy_timeout` from 5s to 30s and added retry-with-backoff on all write paths. Fixes "database is locked" errors when multiple Claude Code sessions share the same `omega.db`.

### Added

- **Entity auto-capture** — `resolve_project_entity()` wired into `bridge.auto_capture()` for automatic entity scoping.
- **Smithery.yaml** — configuration file for Smithery.ai directory listing.
- **Demo GIF** — animated terminal demo in README showing cross-session memory recall.

### Changed

- README restructured: leads with problem statement, demo GIF, and examples section.

## [0.6.0] - 2026-02-13

### Forgetting Intelligence

- **Forgetting Audit Trail** — every deletion logged with reason (TTL, LRU, consolidation, feedback, user). New `omega_forgetting_log` tool to query the log.
- **Decay Curves** — old unaccessed memories rank lower in search results. Preferences and errors exempt. Floor at 0.35.
- **Conflict Detection** — contradictions auto-detected on store. Decisions auto-resolve (newest wins), lessons get flagged for manual review.

26 MCP tools total (+1). 31 new tests for forgetting intelligence features.

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

[0.8.0]: https://github.com/omega-memory/core/releases/tag/v0.8.0
[0.7.3]: https://github.com/omega-memory/core/releases/tag/v0.7.3
[0.7.2]: https://github.com/omega-memory/core/releases/tag/v0.7.2
[0.7.1]: https://github.com/omega-memory/core/releases/tag/v0.7.1
[0.7.0]: https://github.com/omega-memory/core/releases/tag/v0.7.0
[0.6.1]: https://github.com/omega-memory/core/releases/tag/v0.6.1
[0.6.0]: https://github.com/omega-memory/core/releases/tag/v0.6.0
[0.5.0]: https://github.com/omega-memory/core/releases/tag/v0.5.0
