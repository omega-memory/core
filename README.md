# OMEGA

**The memory system for AI coding agents.** Decisions, lessons, and context that persist across sessions.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://github.com/omega-memory/core/actions/workflows/test.yml/badge.svg)](https://github.com/omega-memory/core/actions/workflows/test.yml)
[![PyPI](https://img.shields.io/pypi/v/omega-memory.svg)](https://pypi.org/project/omega-memory/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
Forgetting Intelligence included — memories decay, conflicts auto-resolve, every deletion is audited.

<p align="center">
  <img src="assets/demo.gif" alt="OMEGA demo — cross-session memory recall" width="700">
</p>

---

## Quick Install

```bash
pip install omega-memory

# Set up (creates ~/.omega/, downloads embedding model, registers MCP server)
omega setup

# Verify
omega doctor
```

### From Source

```bash
git clone https://github.com/omega-memory/core.git
cd omega
pip install -e ".[dev]"
omega setup
```

`omega setup` will:
1. Create `~/.omega/` directory
2. Download the ONNX embedding model (~90 MB) to `~/.cache/omega/models/`
3. Register `omega-memory` as an MCP server in `~/.claude.json`
4. Install session hooks in `~/.claude/settings.json`
5. Add a managed `<!-- OMEGA:BEGIN -->` block to `~/.claude/CLAUDE.md`

All changes are idempotent — running `omega setup` again won't duplicate entries.

## The Problem

AI coding agents are stateless. Every new session starts from zero.

- **Context loss.** Agents forget every decision, preference, and architectural choice between sessions. Developers spend 10-30 minutes per session re-explaining context that was already established.
- **Repeated mistakes.** Without learning from past sessions, agents make the same errors over and over. They don't remember what worked, what failed, or why a particular approach was chosen.

OMEGA gives AI coding agents long-term memory and cross-session learning — all running locally on your machine.

## 60-Second Quickstart

OMEGA works through natural language — no API calls, no configuration. Just talk to Claude.

**1. Tell Claude to remember something:**
> "Remember that the auth system uses JWT tokens, not session cookies"

Claude stores this as a permanent memory with semantic embeddings.

**2. Close the session. Open a new one.**

**3. Ask about it:**
> "What did I decide about authentication?"

OMEGA surfaces the relevant memory automatically:
```
Found 1 relevant memory:
  [decision] "The auth system uses JWT tokens, not session cookies"
  Stored 2 days ago | accessed 3 times
```

That's it. Memories persist across sessions, accumulate over time, and are surfaced automatically when relevant — even if you don't explicitly ask.

## Examples

### Architectural Decisions

> "Remember: we chose PostgreSQL over MongoDB for the orders service because we need ACID transactions for payment processing."

Three weeks later, in a new session:

> "I'm adding a caching layer to the orders service — what should I know?"

OMEGA surfaces the PostgreSQL decision automatically, so Claude doesn't suggest a MongoDB-style approach.

### Learning from Mistakes

You spend 30 minutes debugging a Docker build failure. Claude figures it out:

> *"The node_modules volume mount was shadowing the container's node_modules. Fixed by adding an anonymous volume."*

OMEGA auto-captures this as a lesson. Next time anyone hits the same Docker issue, Claude already knows the fix.

### Code Preferences

> "Remember: always use early returns. Never nest conditionals more than 2 levels deep. Prefer `const` over `let`."

Every future session follows these rules without being told again.

### Task Continuity

You're mid-refactor when you need to stop:

> "Checkpoint this — I'm halfway through migrating the auth middleware to the new pattern."

Next session:

> "Resume the auth middleware task."

Claude picks up exactly where you left off — files changed, decisions made, what's left to do.

### Error Patterns

Claude encounters the same `ECONNRESET` three sessions in a row. Each time OMEGA surfaces the previous fix:

```
[error_pattern] ECONNRESET on API calls — caused by connection pool exhaustion.
Fix: set maxSockets to 50 in the http agent config.
Accessed 3 times
```

No more re-debugging the same issue.

## Key Features

- **Persistent Memory** — Stores decisions, lessons, error patterns, and preferences with semantic search. Your agent recalls what matters without you re-explaining everything each session.

- **Semantic Search** — bge-small-en-v1.5 embeddings + sqlite-vec for fast, accurate retrieval. Finds relevant memories even when the wording is different.

- **Cross-Session Learning** — Lessons, preferences, and error patterns accumulate over time. Agents learn from past mistakes and build on previous decisions.

- **Auto-Capture & Surfacing** — Hook system automatically captures important context and surfaces relevant memories before edits, at session start, and during work.

- **Graph Relationships** — Memories are linked with typed edges (related, supersedes, contradicts). Traverse the knowledge graph to find connected context.

- **Encryption at Rest** *(optional)* — AES-256-GCM encrypted storage with macOS Keychain integration. `pip install omega-memory[encrypt]`

- **Forgetting Audit Trail** — Every deletion logged with reason (TTL, LRU, consolidation, feedback, user). Query the log anytime.

- **Decay Curves** — Old unaccessed memories rank lower automatically. Preferences and errors never decay. Floor at 0.35.

- **Conflict Detection** — Contradictions auto-detected on store. Decisions auto-resolve, lessons get flagged.

- **Plugin Architecture** — Extensible via entry points. Add custom tools and handlers through the plugin system.

## How OMEGA Compares

| Feature | OMEGA | Mem0 | Zep | Copilot Memory |
|---------|:-----:|:----:|:---:|:--------------:|
| Local-first (no cloud required) | Yes | No | No | No |
| Semantic search | Yes | Yes | Yes | Limited |
| Cross-session learning | Yes | Limited | No | No |
| Auto-capture & surfacing | Yes | No | No | Partial |
| Graph relationships | Yes | No | No | No |
| Privacy (fully local) | Yes | No | No | No |
| Intelligent forgetting | Yes | No | No | No |
| Free & open source | Yes (Apache-2.0) | Freemium | Freemium | Bundled |

## Open Source

OMEGA is fully open source under Apache-2.0. All 25 memory tools, semantic search, hooks, forgetting intelligence, graph relationships, encryption, and the plugin system are free.

## Compatibility

| Client | Support Level | Setup |
|--------|:------------:|-------|
| Claude Code | Full | `omega setup` (automatic) |
| Cursor | Full | Manual MCP config in settings |
| Windsurf | Full | Manual MCP config in settings |
| Any MCP Client | Full | `claude mcp add omega-memory -- python3 -m omega.server.mcp_server` |

Requires Python 3.11+. macOS and Linux supported. Windows via WSL.

## Architecture

```
               ┌─────────────────────┐
               │    Claude Code       │
               │  (or any MCP host)   │
               └──────────┬──────────┘
                          │ stdio/MCP
               ┌──────────▼──────────┐
               │   OMEGA MCP Server   │
               │   25 memory tools    │
               └──────────┬──────────┘
                          │
               ┌──────────▼──────────┐
               │    omega.db (SQLite) │
               │ memories | edges |   │
               │     embeddings       │
               └──────────────────────┘
```

Single database, modular handlers. Additional tools available via the plugin system. No separate daemons, no microservices.

## MCP Tools Reference

OMEGA runs as an MCP server inside Claude Code. Once installed, 25 memory tools are available. Additional tools can be added via the plugin system.

| Tool | What it does |
|------|-------------|
| `omega_remember` | Store a permanent memory ("remember this") |
| `omega_store` | Store typed memory (decision, lesson, error, summary) |
| `omega_query` | Semantic search with tag filters and contextual re-ranking |
| `omega_phrase_search` | Exact phrase search via FTS5 |
| `omega_lessons` | Cross-session lessons ranked by access count |
| `omega_welcome` | Session briefing with recent memories and profile |
| `omega_compact` | Cluster and summarize related memories |
| `omega_consolidate` | Prune stale memories, cap summaries, clean edges |
| `omega_timeline` | Memories grouped by day |
| `omega_similar` | Find memories similar to a given one |
| `omega_traverse` | Walk the relationship graph |
| `omega_checkpoint` | Save task state for cross-session continuity |
| `omega_resume_task` | Resume a previously checkpointed task |
| `omega_forgetting_log` | Query the forgetting audit trail (deletions with reasons) |

## CLI

| Command | Description |
|---------|-------------|
| `omega setup` | Create dirs, download model, register MCP, install hooks |
| `omega doctor` | Verify installation health |
| `omega status` | Memory count, store size, model status |
| `omega query <text>` | Search memories by semantic similarity |
| `omega store <text>` | Store a memory with a specified type |
| `omega timeline` | Show memory timeline grouped by day |
| `omega activity` | Show recent session activity overview |
| `omega stats` | Memory type distribution and health summary |
| `omega consolidate` | Deduplicate, prune, and optimize memory |
| `omega compact` | Cluster and summarize related memories |
| `omega backup` | Back up omega.db (keeps last 5) |
| `omega validate` | Validate database integrity |
| `omega logs` | Show recent hook errors |
| `omega migrate-db` | Migrate legacy JSON to SQLite |

<details>
<summary><strong>Advanced Details</strong></summary>

### Hooks (4 hooks, 4 handlers)

All hooks dispatch via `fast_hook.py` → daemon UDS socket, with fail-open semantics.

| Hook | Handlers | Purpose |
|------|----------|---------|
| SessionStart | `session_start` | Welcome briefing with recent memories |
| Stop | `session_stop` | Session summary |
| UserPromptSubmit | `auto_capture` | Auto-capture lessons/decisions |
| PostToolUse | `surface_memories` | Surface relevant memories during work |

### Storage

| Path | Purpose |
|------|---------|
| `~/.omega/omega.db` | SQLite database (memories, embeddings, edges) |
| `~/.omega/profile.json` | User profile |
| `~/.omega/hooks.log` | Hook error log |
| `~/.cache/omega/models/bge-small-en-v1.5-onnx/` | ONNX embedding model |

### Search Pipeline

1. **Vector similarity** via sqlite-vec (cosine distance, 384-dim bge-small-en-v1.5)
2. **Full-text search** via FTS5 (fast keyword matching)
3. **Type-weighted scoring** (decisions/lessons weighted 2x)
4. **Contextual re-ranking** (boosts by tag, project, and content match)
5. **Deduplication** at query time
6. **Time-decay weighting** (old unaccessed memories rank lower)

### Memory Lifecycle

- **Dedup**: SHA256 hash (exact) + embedding similarity 0.85+ (semantic) + Jaccard per-type
- **Evolution**: Similar content (55-95%) appends new insights to existing memories
- **TTL**: Session summaries expire after 1 day, lessons/preferences are permanent
- **Auto-relate**: Creates `related` edges (similarity >= 0.45) to top-3 similar memories
- **Compaction**: Clusters and summarizes related memories
- **Decay**: Unaccessed memories lose ranking weight over time (floor 0.35); preferences and errors exempt
- **Conflict detection**: Contradicting memories auto-detected on store; decisions auto-resolve, lessons flagged

### Memory Footprint

- Startup: ~31 MB RSS
- After first query (ONNX model loaded): ~337 MB RSS
- Database: ~10.5 MB for ~242 memories

</details>

## Troubleshooting

**`omega doctor` shows FAIL on import:**
- Ensure `pip install -e .` from the repo root
- Check `python3 -c "import omega"` works

**MCP server not registered:**
```bash
claude mcp add omega-memory -- python3 -m omega.server.mcp_server
```

**Hooks not firing:**
- Check `~/.claude/settings.json` has OMEGA hook entries
- Check `~/.omega/hooks.log` for errors

## Development

```bash
pip install -e ".[dev]"
pytest tests/
ruff check src/              # Lint
```

## Uninstall

```bash
claude mcp remove omega-memory
rm -rf ~/.omega ~/.cache/omega
pip uninstall omega-memory
```

Manually remove OMEGA entries from `~/.claude/settings.json` and the `<!-- OMEGA:BEGIN -->` block from `~/.claude/CLAUDE.md`.

## Contributing

- [Contributing Guide](CONTRIBUTING.md)
- [Security Policy](SECURITY.md)
- [Changelog](CHANGELOG.md)
- [Report a Bug](https://github.com/omega-memory/core/issues)

## License

Apache-2.0 — see [LICENSE](LICENSE) for details.
