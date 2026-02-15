# Reddit Post Drafts

## Post 1: r/ClaudeAI

**Title:** I gave Claude Code persistent memory across sessions — it actually remembers decisions and learns from mistakes now

**Body:**

I've been using Claude Code as my daily driver for about 6 months. The biggest friction point for me was context loss — every new session starts from zero. I'd spend 10-15 minutes re-explaining architecture decisions, code preferences, and past debugging sessions.

So I built a memory system that runs locally via MCP. It auto-captures decisions and lessons during coding sessions, then surfaces them when they're relevant again. A few examples of what changed:

**Before:** "We need to add caching to the orders service." Claude suggests Redis. I explain again that we specifically chose PostgreSQL for this service because we need ACID transactions. Again.

**After:** Claude starts the session already knowing about the PostgreSQL decision because OMEGA surfaced it automatically.

**Before:** Same Docker volume mount bug three sessions in a row. Debug from scratch each time.

**After:** Claude gets the fix surfaced before I even describe the error.

The system also supports checkpointing — I can stop mid-refactor, come back the next day, and Claude picks up exactly where I left off (files changed, decisions made, what's left to do).

Some technical details if you're curious:
- Runs entirely locally — SQLite + ONNX embeddings, no cloud, no API keys
- 25 MCP tools for storing, querying, and managing memories
- Auto-capture works through Claude Code's hook system (SessionStart, PostToolUse, UserPromptSubmit)
- Memories decay over time so the context stays relevant — but preferences and error patterns are permanent
- Scored 95.4% on LongMemEval (#1 on the leaderboard — an academic benchmark for long-term memory, 500 questions)

Install is just `pip install omega-memory && omega setup` — it auto-configures Claude Code's MCP and hooks.

It's open source (Apache-2.0): https://github.com/omega-memory/omega-memory

Happy to answer questions about how the memory capture works or the MCP integration. I'm still iterating on the auto-capture prompts and would love feedback from other Claude Code users on what types of context you wish persisted between sessions.

---

## Post 2: r/mcp

**Title:** Lessons from building a 25-tool MCP server for persistent agent memory

**Body:**

I've been building an MCP server that gives AI coding agents persistent memory across sessions. Wanted to share some things I learned about MCP server architecture along the way, since this sub has been helpful.

**The problem:** Claude Code, Cursor, Windsurf — they're all stateless. Every session starts fresh. My server exposes 25 memory tools via MCP so agents can store, query, checkpoint, and manage long-term memory.

**Architecture decisions that mattered:**

1. **SQLite over Postgres/Neo4j.** Controversial maybe, but for a local-first tool this was the right call. Zero setup, single file, ships everywhere. I use sqlite-vec for vector similarity and FTS5 for keyword search. The tradeoff: no built-in graph database, so I model relationships as typed edges in a regular table (related, supersedes, contradicts). It works well enough for the scale I'm targeting.

2. **ONNX embeddings over API calls.** I bundle bge-small-en-v1.5 as an ONNX model (~90 MB download, runs on CPU). This means zero network dependencies — no OpenAI key, no Ollama, nothing to configure. The tradeoff: English-only and 384-dim vectors. Good enough for code context and natural language queries, but won't handle multilingual use cases.

3. **Tool granularity.** I started with 5 big tools (store, query, manage, checkpoint, system) but found that agents work better with more specific, single-purpose tools. Ended up with 25 — things like `omega_lessons` (just lessons learned), `omega_similar` (find related memories), `omega_traverse` (walk the relationship graph). Agents are surprisingly good at picking the right tool when the names are descriptive.

4. **Hook-based auto-capture.** The highest-value feature isn't any MCP tool — it's the Claude Code hook system that auto-captures decisions and lessons without the user explicitly asking. I hook into SessionStart (for welcome briefings), PostToolUse (surface relevant memories after file edits), and UserPromptSubmit (detect decisions/lessons in conversation). This only works in Claude Code right now since other clients don't have hooks.

5. **Intelligent forgetting.** Every memory system eventually drowns in noise. I implemented: TTL-based expiry (session summaries expire in 1 day, lessons are permanent), consolidation (merge similar memories), compaction (cluster and summarize), conflict detection (new decision contradicts old one), and time-decay (unaccessed memories rank lower). Every deletion is audited — you can query why any memory was removed.

**Search pipeline** (this took the most iteration):
Vector similarity → FTS5 keyword → type-weighted scoring (decisions 2x) → contextual re-ranking (project, tags, content match) → time-decay → dedup

The re-ranking step matters a lot. Pure vector similarity returns semantically similar but contextually irrelevant results. Boosting by current project and recently-touched files made a huge difference.

**Benchmark:** 95.4% (466/500, task-averaged) on LongMemEval (ICLR 2025), #1 on the leaderboard. The benchmark tests 5 categories: extraction, multi-session reasoning, temporal reasoning, knowledge updates, and preference tracking. The hardest category was multi-session reasoning — counting and aggregating across many sessions.

Open source (Apache-2.0): https://github.com/omega-memory/omega-memory

Curious what other MCP server builders have found regarding tool granularity — do agents in your experience work better with fewer fat tools or many focused ones?
