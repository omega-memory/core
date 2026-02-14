# Changelog

All notable changes to OMEGA will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

[0.6.1]: https://github.com/omega-memory/core/releases/tag/v0.6.1
[0.6.0]: https://github.com/omega-memory/core/releases/tag/v0.6.0
[0.5.0]: https://github.com/omega-memory/core/releases/tag/v0.5.0
