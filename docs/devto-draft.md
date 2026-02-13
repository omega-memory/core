---
title: How I Built a Memory System That Scores 95.4% on LongMemEval (#1 on the Leaderboard)
published: false
description: A solo builder's deep dive into OMEGA — a local-first persistent memory system for AI coding agents. Architecture, benchmark methodology, and honest tradeoffs.
tags: ai, opensource, python, machinelearning
cover_image:
canonical_url: https://omegamax.co
---

# How I Built a Memory System That Scores 95.4% on LongMemEval (#1 on the Leaderboard)

Every AI coding agent has the same problem: amnesia.

You spend an hour with Claude Code debugging a Docker volume mount issue. You find the fix, explain your architectural reasoning, set coding preferences. Then you close the session. Next time you open it, the agent has no idea any of that happened. You start from zero.

I got tired of spending the first 10-15 minutes of every session re-explaining context that was already established. So I built OMEGA — a persistent memory system that gives AI coding agents long-term memory across sessions. It runs entirely on your machine, scores 95.4% on the LongMemEval academic benchmark (#1 on the leaderboard), and you can install it with `pip install omega-memory`.

This post is a technical walkthrough of what I built, how it works, and where it falls short.

## The Problem, Concretely

AI coding agents today are stateless by design. The conversation context is the only "memory" they have, and it evaporates when the session ends.

This means:

- **Decisions vanish.** "We chose PostgreSQL over MongoDB because we need ACID transactions for payment processing" — gone. Next session, the agent might suggest MongoDB.
- **Mistakes repeat.** You debug the same `ECONNRESET` error three sessions in a row because the agent doesn't remember it was caused by connection pool exhaustion.
- **Preferences reset.** "Always use early returns, never nest conditionals more than 2 levels deep" — you have to say this every single time.

The root cause is that MCP (Model Context Protocol) gives agents access to tools, but no persistent state between sessions. There's no standard way for an agent to store and recall what it learned.

## Architecture: SQLite All the Way Down

I went through a few iterations. The first version used an in-memory graph (NetworkX). At around 3,700 nodes it consumed 372 MB of RAM. That was unacceptable for something that runs in the background.

The current architecture is much simpler:

```
┌─────────────────────────┐
│    Claude Code / Cursor  │
│    (any MCP host)        │
└───────────┬─────────────┘
            │ stdio/MCP protocol
┌───────────▼─────────────┐
│   OMEGA MCP Server       │
│   26 memory tools        │
│                          │
│  ┌─────────────────────┐ │
│  │  Hook Daemon (UDS)  │ │    ← Unix Domain Socket for
│  │  auto-capture +     │ │      <750ms hook dispatch
│  │  auto-surface       │ │
│  └─────────────────────┘ │
└───────────┬─────────────┘
            │
┌───────────▼─────────────┐
│  omega.db (SQLite + WAL) │
│                          │
│  ┌──────┐ ┌───────────┐ │
│  │nodes │ │ vec_nodes  │ │    ← sqlite-vec: 384-dim
│  │      │ │ (vectors)  │ │      cosine similarity
│  └──────┘ └───────────┘ │
│  ┌──────┐ ┌───────────┐ │
│  │edges │ │ nodes_fts  │ │    ← FTS5: full-text
│  │      │ │ (keywords) │ │      keyword search
│  └──────┘ └───────────┘ │
└─────────────────────────┘
            │
┌───────────▼─────────────┐
│  bge-small-en-v1.5       │
│  ONNX Runtime (CPU)      │
│  384-dim embeddings      │
│  ~90 MB model on disk    │
└─────────────────────────┘
```

Everything is a single SQLite database (`~/.omega/omega.db`) running in WAL mode with `sqlite-vec` for vector search and FTS5 for keyword matching. The embedding model is bge-small-en-v1.5 running via ONNX Runtime on CPU — no GPU required, no cloud API calls.

**Why SQLite?** Because the access pattern is perfect for it. One machine, one user, mostly reads with occasional writes, and the entire database fits in a few megabytes. At ~250 memories, the database is about 10 MB. SQLite's WAL mode handles concurrent reads from multiple MCP server processes, and I added retry-with-backoff for the rare write contention under heavy multi-process usage.

**Why not a vector database?** I considered Chroma and Qdrant. But adding a separate database process for a system that stores hundreds (not millions) of vectors felt like overengineering. `sqlite-vec` gives me cosine similarity search in the same process, with zero external dependencies.

## The Search Pipeline

Retrieval accuracy is everything for a memory system. If you can't find the right memory when it matters, the whole system is useless. I landed on a six-stage pipeline:

```
Query: "Docker volume mount issue"
           │
           ▼
┌─────────────────────────┐
│ 1. Vector Similarity    │  cosine distance on 384-dim
│    (sqlite-vec)         │  embeddings, top-K candidates
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│ 2. Full-Text Search     │  FTS5 keyword matching for
│    (FTS5)               │  terms the embeddings miss
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│ 3. Type-Weighted Score  │  decisions and lessons get 2x
│                         │  weight (they're higher value)
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│ 4. Contextual Re-rank   │  boost by tag match, project
│                         │  scope, and content overlap
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│ 5. Time-Decay           │  old unaccessed memories rank
│                         │  lower (floor 0.35, exemptions
│                         │  for prefs + error patterns)
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│ 6. Dedup                │  remove near-duplicates from
│                         │  the result set
└───────────┬─────────────┘
            ▼
        Results
```

The two-source approach (vectors + FTS5) is key. Embeddings handle semantic similarity ("container networking issue" matches "Docker bridge network problem"), while FTS5 catches exact terms that embeddings sometimes miss (specific error codes, package names, config keys). Combining both gives better recall than either alone.

Type-weighted scoring was a deliberate design choice. When you ask "what should I know about the orders service?", a prior architectural decision ("we chose PostgreSQL for ACID compliance") is almost always more relevant than a session summary from three weeks ago. Weighting decisions and lessons 2x in the scoring reflects this.

## Memory Lifecycle: Why Forgetting Matters

Most memory systems just accumulate. Store everything, search through everything, forever. This works at 50 memories. At 500, you start getting noise in every query. At 5,000, the system is actively harmful — surfacing outdated context that leads the agent astray.

OMEGA has an explicit forgetting system with five mechanisms:

1. **Dedup on write.** SHA-256 hash for exact duplicates, plus embedding similarity (threshold 0.85) for semantic duplicates. If you store "use PostgreSQL for the orders DB" twice with different wording, OMEGA catches it.

2. **Evolution.** When new content is 55-95% similar to an existing memory, instead of creating a new entry, OMEGA appends the new information to the existing one. The memory evolves rather than duplicates.

3. **TTL expiry.** Session summaries expire after 1 day — they're useful for immediate context but stale quickly. Lessons and preferences are permanent. Everything else gets a configurable TTL.

4. **Compaction.** Periodically, OMEGA clusters related memories (Jaccard similarity) and summarizes them into consolidated nodes, marking the originals as superseded. This is like garbage collection for knowledge.

5. **Conflict detection.** When a new decision contradicts an existing one, OMEGA detects it automatically. Decisions auto-resolve (newer wins), while lessons are flagged for review. The old memory gets a `contradicts` edge to the new one.

Every deletion is audited. You can run `omega_forgetting_log` and see exactly why each memory was removed — TTL expired, consolidation pruned, compaction superseded, LRU evicted, user deleted, or flagged via feedback.

## Auto-Capture: The Part That Actually Matters

The 26 MCP tools are nice, but the real value is in the hook system. In Claude Code, OMEGA installs four hooks:

| Hook Event | What It Does |
|---|---|
| **SessionStart** | Surfaces relevant memories as a welcome briefing |
| **PostToolUse** | After file edits, surfaces memories related to the file being changed |
| **UserPromptSubmit** | Analyzes the conversation and auto-captures decisions/lessons |
| **Stop** | Generates a session summary |

The hooks dispatch through a Unix Domain Socket daemon running inside the MCP server process. This is important — the first version spawned a new Python process per hook invocation, which added ~750ms of cold-start overhead. The UDS daemon eliminates that by reusing the warm MCP server with its already-loaded ONNX model and database connection.

The `UserPromptSubmit` hook is the most interesting. It classifies the conversation context and extracts decisions ("we chose X because Y"), lessons ("the fix for X is Y"), and error patterns — all without the user explicitly saying "remember this." You just code normally, and OMEGA captures what matters in the background.

## Benchmarking Against LongMemEval

[LongMemEval](https://github.com/xiaowu0162/LongMemEval) is an academic benchmark from ICLR 2025 that tests long-term memory systems with 500 questions across 5 capabilities:

- **Information Extraction (IE):** Can you recall specific facts from past conversations?
- **Multi-Session Reasoning (MS):** Can you synthesize information across multiple sessions?
- **Temporal Reasoning (TR):** Can you reason about when things happened and in what order?
- **Knowledge Update (KU):** When information changes, do you return the current state?
- **Preference Tracking (SP):** Do you remember and apply user preferences?

Here's how OMEGA stacks up:

| System | Overall | IE | MS | TR | KU | SP |
|--------|--------:|---:|---:|---:|---:|---:|
| **OMEGA** | **95.4%** | 100% | 83.5% | 94.0% | 96.2% | 98.6% |
| Mastra | 94.87% | — | — | — | — | — |
| Emergence | 86.0% | — | — | — | — | — |
| Zep/Graphiti | 71.2% | — | — | — | — | — |

A few notes on methodology and honesty:

**What the benchmark does well:** It tests real conversational memory patterns — things like "what restaurant did I mention in our 3rd conversation?" or "I changed my coffee preference from latte to americano, what's my current preference?" These are practical tests of what a memory system needs to handle.

**What it doesn't test:** It doesn't test auto-capture quality, retrieval latency under load, or how well the system handles adversarial/contradictory inputs over time. It also doesn't test multi-agent coordination, which is a significant part of OMEGA's feature set.

**Where OMEGA struggles:** Multi-session reasoning (83.5%) is the weakest category. These questions require counting or aggregating across many sessions ("how many times did I mention going to the gym?"), which is fundamentally harder for a retrieval-based system. My best result came from a simple "list all matches, then count" approach — more aggressive dedup strategies actually caused regressions.

**Scoring methodology:** The 95.4% is task-averaged — the mean of per-category accuracies. This is the same methodology used by other systems on the leaderboard (including Mastra at 94.87%). The raw score is 466/500 (93.2%).

**Cost of benchmarking:** Each full run costs real money in LLM API calls (the benchmark uses GPT-4 for evaluation). I ran about 8 iterations to get from 85% to 95.4%, each time targeting specific failure modes. The improvements were incremental: better temporal prompting (+5 questions), knowledge-update current-state prompting (+4), query augmentation (+2), preference personalization (+2).

## The Competition

I want to be fair here, because these are all legitimate projects solving the same problem from different angles:

- **Mem0** (47K GitHub stars): Cloud-first approach. More polished product, larger team, established user base. Requires an API key for the cloud version. Their local mode exists but is more limited. They haven't published LongMemEval scores.

- **Zep/Graphiti** (22.8K stars): Neo4j-backed knowledge graph approach. Sophisticated architecture but requires running Neo4j. Published 71.2% on LongMemEval in their paper, which I respect — most systems don't publish benchmark numbers at all.

- **Letta** (21.1K stars): Agent framework with memory as a component. Different scope — they're building a full agent platform, not just memory.

- **Claude's built-in memory** (CLAUDE.md files): Works without any setup, but it's a flat markdown file with no semantic search, no auto-capture, and no cross-session learning.

OMEGA's differentiator is being local-first with zero external dependencies while still scoring competitively on benchmarks. Whether that tradeoff matters to you depends on your threat model and workflow.

## Honest Tradeoffs and Limitations

I'd rather you know the sharp edges before you try it:

- **Auto-capture hooks only work with Claude Code.** Other MCP clients (Cursor, Windsurf, Zed) get the 26 tools but not the automatic memory capture. You have to explicitly tell the agent to remember things.

- **Memory footprint.** ~31 MB at startup, ~337 MB after the ONNX model loads on first query. The model unloads after 10 minutes of inactivity, but if you're memory-constrained, this matters.

- **English only.** The bge-small-en-v1.5 embedding model is trained on English text. It will work poorly for other languages.

- **Solo maintainer.** This is a passion project, not a VC-backed company. I maintain it because I use it every day, but I can't promise the same velocity as a funded team.

- **Not stress-tested at scale.** I've been running it at ~250 memories with no issues. I haven't tested at 10K+ memories. SQLite can handle it, but the search pipeline might need tuning at that scale.

- **Python 3.11+ only.** No support for older Python versions. macOS and Linux are supported; Windows works through WSL.

## Try It

```bash
pip install omega-memory
omega setup          # auto-detects Claude Code
omega doctor         # verify everything works
```

For Cursor or Windsurf:

```bash
omega setup --client cursor
omega setup --client windsurf
```

Three commands, no API keys, no cloud accounts, no Docker containers. Everything runs locally.

The source is at [github.com/omega-memory/core](https://github.com/omega-memory/core) under Apache-2.0. Stars are appreciated — the project has about 5 right now. Contributions, bug reports, and questions are welcome.

If you want to see the benchmark methodology in detail or how OMEGA compares to alternatives with sources, check [omegamax.co](https://omegamax.co).

---

*I build OMEGA because I use it every day. The best test of a developer tool is whether the developer actually uses it — and I haven't opened a Claude Code session without OMEGA in months. If you're spending time re-explaining context to your AI coding agent, give it a try.*
