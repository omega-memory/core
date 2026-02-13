# Show HN Draft

**Title:** Show HN: OMEGA – Local-first memory for AI coding agents (95.4% on LongMemEval, #1 on the leaderboard)

**URL:** https://github.com/omega-memory/core

---

**Post text:**

I built OMEGA because I was tired of re-explaining context to Claude Code every session.

Every time I start a new session, the agent has no idea that we chose PostgreSQL over MongoDB last week, that the Docker volume mount issue was already debugged, or that I prefer early returns over nested conditionals. I'd spend the first 10-15 minutes of each session re-establishing context that was already figured out.

OMEGA is a persistent memory system that runs locally on your machine. It captures decisions, lessons, and error patterns during coding sessions, then surfaces them when they're relevant again. It works through MCP (Model Context Protocol), so it plugs into Claude Code, Cursor, Windsurf, or any MCP-compatible client.

**How it works:**

- SQLite database for storage (~10 MB for ~250 memories)
- bge-small-en-v1.5 ONNX embeddings for semantic search (CPU-only, no GPU needed)
- 26 MCP tools exposed to the agent (store, query, checkpoint, resume, etc.)
- Hook system in Claude Code auto-captures lessons and decisions without explicit commands
- Memories are linked via typed edges (related, supersedes, contradicts) forming a knowledge graph

The search pipeline is: vector similarity → FTS5 keyword matching → type-weighted scoring → contextual re-ranking → time-decay → dedup. Decisions and lessons are weighted 2x because they're more valuable than general notes.

**What makes it different from Mem0/Zep/Letta:**

The main difference is architectural. Mem0 requires an API key and cloud. Zep needs Neo4j. OMEGA is SQLite + a bundled ONNX model — `pip install omega-memory && omega setup` and you're running. No accounts, no cloud, no external services. Everything stays on your machine.

The other differentiator is intelligent forgetting. Most memory systems just accumulate. OMEGA has TTL-based expiry, consolidation (merge similar memories), compaction (cluster and summarize), and conflict detection (when a new decision contradicts an old one). Every deletion is audited so you can see why a memory was removed.

**Benchmark:**

I ran OMEGA against LongMemEval (ICLR 2025), an academic benchmark with 500 questions testing long-term memory across 5 capabilities: information extraction, multi-session reasoning, temporal reasoning, knowledge updates, and preference tracking.

OMEGA scored 95.4% (466/500, task-averaged), which puts it #1 on the leaderboard — ahead of Mastra (94.87%). Task-averaged scoring (mean of per-category accuracies) is the standard methodology used by other systems on the leaderboard. Zep/Graphiti published 71.2% in their paper. Most other systems haven't published scores.

The benchmark code and methodology are open — anyone can reproduce it.

**Honest tradeoffs:**

- Auto-capture hooks only work with Claude Code right now. Other MCP clients get the 26 tools but not the automatic memory capture.
- Memory footprint is ~31 MB at startup, ~337 MB after the ONNX model loads on first query.
- The embedding model (bge-small-en-v1.5) is English-only.
- At ~250 memories the database is ~10 MB. I haven't stress-tested at 10K+ memories yet.
- Solo maintainer. This is a passion project, not a VC-backed startup.

**Install:**

```
pip install omega-memory
omega setup
omega doctor
```

Apache-2.0. Python 3.11+. macOS and Linux (Windows via WSL).

Happy to answer questions about the architecture, benchmark methodology, or anything else.
