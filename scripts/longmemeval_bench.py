#!/usr/bin/env python3
"""
LongMemEval-inspired benchmark for OMEGA retrieval quality.

Measures retrieval accuracy across 5 categories inspired by the LongMemEval
benchmark (Wang et al., 2024):
  - Information extraction (IE): Can OMEGA find specific facts?
  - Multi-session reasoning (MS): Can OMEGA connect info across sessions?
  - Temporal reasoning (TR): Can OMEGA handle time-based queries?
  - Knowledge update (KU): Does OMEGA prefer updated info over stale?
  - Abstention (AB): Does OMEGA avoid surfacing irrelevant results?

Scores: "Did the correct memory appear in top-3 results?"
No LLM API calls — purely local, free, fast, repeatable.

Usage:
    python scripts/longmemeval_bench.py [--verbose]
"""

import argparse
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Ensure omega is importable
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def create_bench_store(db_path: str):
    """Create a fresh SQLiteStore for benchmarking."""
    import os
    os.environ["OMEGA_HOME"] = str(Path(db_path).parent)
    from omega.sqlite_store import SQLiteStore
    return SQLiteStore(db_path=db_path)


def seed_memories(store):
    """Seed ~100 synthetic memories across 5 categories."""
    now = datetime.now(timezone.utc)
    memories = []

    # === Information Extraction (20 memories) ===
    ie_data = [
        ("Python's GIL prevents true parallel execution of CPU-bound threads. Use multiprocessing for CPU parallelism.",
         {"event_type": "lesson_learned", "priority": 4, "tags": ["python", "threading"]}),
        ("SQLite WAL mode allows concurrent reads while writing. Set PRAGMA journal_mode=WAL on connection.",
         {"event_type": "decision", "priority": 4, "tags": ["sqlite"]}),
        ("ONNX Runtime CPU inference uses ~337MB RAM for bge-small-en-v1.5 model.",
         {"event_type": "lesson_learned", "priority": 4, "tags": ["onnx", "embedding"]}),
        ("React useEffect cleanup function runs before each re-execution and on unmount.",
         {"event_type": "lesson_learned", "priority": 4, "tags": ["react", "javascript"]}),
        ("Docker layer caching: put rarely-changing instructions (apt-get) before frequently-changing ones (COPY .).",
         {"event_type": "lesson_learned", "priority": 4, "tags": ["docker"]}),
        ("PostgreSQL VACUUM ANALYZE should be scheduled weekly for tables with heavy UPDATE/DELETE traffic.",
         {"event_type": "decision", "priority": 3, "tags": ["postgres"]}),
        ("TypeScript discriminated unions use a literal type field to narrow union members in switch statements.",
         {"event_type": "lesson_learned", "priority": 4, "tags": ["typescript"]}),
        ("Redis SCAN is preferred over KEYS in production — KEYS blocks the single-threaded event loop.",
         {"event_type": "error_pattern", "priority": 4, "tags": ["redis"]}),
        ("Git rebase --onto allows moving a branch from one base to another without replaying all commits.",
         {"event_type": "lesson_learned", "priority": 4, "tags": ["git"]}),
        ("Kubernetes liveness probes should NOT check dependencies. Readiness probes should.",
         {"event_type": "lesson_learned", "priority": 4, "tags": ["kubernetes"]}),
        ("The user prefers Tailwind CSS over styled-components for styling React applications.",
         {"event_type": "user_preference", "priority": 5, "tags": ["react", "tailwind", "css"]}),
        ("Always use parameterized queries to prevent SQL injection — never concatenate user input.",
         {"event_type": "lesson_learned", "priority": 4, "tags": ["sql", "security"]}),
        ("Python asyncio.gather() runs coroutines concurrently. Use return_exceptions=True to avoid one failure cancelling others.",
         {"event_type": "lesson_learned", "priority": 4, "tags": ["python", "async"]}),
        ("Nginx proxy_pass with trailing slash strips the location prefix from the proxied URL.",
         {"event_type": "error_pattern", "priority": 4, "tags": ["nginx"]}),
        ("JWT tokens should be short-lived (15 min) with refresh tokens for session continuity.",
         {"event_type": "decision", "priority": 4, "tags": ["auth", "security"]}),
        ("The user's timezone is America/New_York (EST/EDT).",
         {"event_type": "user_preference", "priority": 5, "tags": ["profile"]}),
        ("Rust's borrow checker prevents data races at compile time. Use Arc<Mutex<T>> for shared mutable state across threads.",
         {"event_type": "lesson_learned", "priority": 4, "tags": ["rust"]}),
        ("Next.js App Router uses React Server Components by default. Add 'use client' directive for client-side interactivity.",
         {"event_type": "lesson_learned", "priority": 4, "tags": ["next.js", "react"]}),
        ("CoreML causes memory leaks on Apple Silicon when loading models repeatedly. Use ONNX CPU backend instead.",
         {"event_type": "error_pattern", "priority": 4, "tags": ["onnx", "apple"]}),
        ("The project uses Zustand for state management instead of Redux — simpler API, less boilerplate.",
         {"event_type": "decision", "priority": 4, "tags": ["zustand", "react"]}),
    ]

    for content, meta in ie_data:
        meta["session_id"] = "bench-ie"
        memories.append({"content": content, "metadata": meta})

    # === Multi-session reasoning (20 memories across 4 sessions) ===
    ms_sessions = ["bench-ms-1", "bench-ms-2", "bench-ms-3", "bench-ms-4"]
    ms_data = [
        ("Decided to use SQLite for the OMEGA backend because it's zero-config and embedded.", "bench-ms-1",
         {"event_type": "decision", "priority": 4, "tags": ["sqlite", "omega"]}),
        ("Added sqlite-vec extension for vector similarity search in OMEGA.", "bench-ms-2",
         {"event_type": "task_completion", "priority": 3, "tags": ["sqlite", "omega"]}),
        ("FTS5 full-text search index added to OMEGA for fast keyword queries.", "bench-ms-3",
         {"event_type": "task_completion", "priority": 3, "tags": ["sqlite", "omega"]}),
        ("OMEGA schema migration system uses ALTER TABLE for backwards-compatible upgrades.", "bench-ms-4",
         {"event_type": "decision", "priority": 4, "tags": ["sqlite", "omega"]}),
        ("The API server uses FastAPI with uvicorn for the HTTP layer.", "bench-ms-1",
         {"event_type": "decision", "priority": 4, "tags": ["fastapi", "python"]}),
        ("Added rate limiting middleware to the API — 100 req/min per IP.", "bench-ms-2",
         {"event_type": "task_completion", "priority": 3, "tags": ["fastapi", "security"]}),
        ("Switched from JSON file storage to PostgreSQL for the main application database.", "bench-ms-1",
         {"event_type": "decision", "priority": 4, "tags": ["postgres"]}),
        ("Added connection pooling with pgbouncer to handle concurrent database connections.", "bench-ms-3",
         {"event_type": "task_completion", "priority": 3, "tags": ["postgres"]}),
        ("Implemented retry logic with exponential backoff for external API calls.", "bench-ms-2",
         {"event_type": "lesson_learned", "priority": 4, "tags": ["python", "api"]}),
        ("The retry decorator uses jitter to prevent thundering herd in distributed systems.", "bench-ms-4",
         {"event_type": "lesson_learned", "priority": 4, "tags": ["python", "distributed"]}),
        ("Deployed the application to AWS ECS with Fargate for serverless container management.", "bench-ms-1",
         {"event_type": "task_completion", "priority": 3, "tags": ["aws", "docker"]}),
        ("Added CloudWatch alarms for CPU > 80% and memory > 90% on ECS tasks.", "bench-ms-3",
         {"event_type": "decision", "priority": 4, "tags": ["aws", "monitoring"]}),
        ("The CI/CD pipeline uses GitHub Actions with separate staging and production workflows.", "bench-ms-2",
         {"event_type": "decision", "priority": 4, "tags": ["git"]}),
        ("Added automated database migration step to the CI/CD pipeline before deployment.", "bench-ms-4",
         {"event_type": "task_completion", "priority": 3, "tags": ["git"]}),
        ("Implemented feature flags using LaunchDarkly for gradual rollouts.", "bench-ms-1",
         {"event_type": "decision", "priority": 4}),
        ("Feature flags reduced deployment risk — can disable new features without redeploying.", "bench-ms-3",
         {"event_type": "lesson_learned", "priority": 4}),
        ("The user authentication flow uses OAuth 2.0 with Google and GitHub providers.", "bench-ms-2",
         {"event_type": "decision", "priority": 4, "tags": ["auth"]}),
        ("Added PKCE flow for the OAuth implementation to prevent authorization code interception.", "bench-ms-4",
         {"event_type": "task_completion", "priority": 3, "tags": ["auth", "security"]}),
        ("Monitoring stack uses Prometheus for metrics and Grafana for dashboards.", "bench-ms-1",
         {"event_type": "decision", "priority": 4}),
        ("Added custom Prometheus metrics for business KPIs: user signups, API latency p99.", "bench-ms-3",
         {"event_type": "task_completion", "priority": 3}),
    ]

    for content, sid, meta in ms_data:
        meta["session_id"] = sid
        memories.append({"content": content, "metadata": meta})

    # === Temporal reasoning (20 memories with referenced_date) ===
    for i in range(20):
        days_ago = i * 3  # Spread across 60 days
        ref_date = (now - timedelta(days=days_ago)).isoformat()
        content = f"Sprint {20-i} completed: deployed feature batch #{20-i} to production on {(now - timedelta(days=days_ago)).strftime('%Y-%m-%d')}."
        meta = {
            "event_type": "task_completion",
            "priority": 3,
            "session_id": f"bench-tr-{i}",
            "referenced_date": ref_date,
            "tags": ["sprint"],
        }
        memories.append({"content": content, "metadata": meta})

    # === Knowledge update (20 memories — pairs of old + new) ===
    ku_pairs = [
        ("The API response format uses XML for all endpoints.", "The API response format was migrated from XML to JSON for all endpoints."),
        ("Database backups run daily at 2 AM UTC.", "Database backups now run every 6 hours (4x daily) after the data loss incident."),
        ("The frontend uses Create React App for build tooling.", "Migrated from Create React App to Vite for 10x faster builds."),
        ("Authentication tokens expire after 24 hours.", "Authentication tokens now expire after 15 minutes with refresh token rotation."),
        ("The application runs on a single EC2 instance.", "Migrated from single EC2 to ECS Fargate with auto-scaling (2-10 tasks)."),
        ("Logging uses console.log statements throughout the codebase.", "Implemented structured logging with Winston — all console.log replaced."),
        ("Tests run manually before each deployment.", "CI/CD pipeline runs tests automatically on every pull request."),
        ("The database schema has no migration system.", "Added Alembic for database schema migrations with version tracking."),
        ("Error handling returns generic 500 responses.", "Implemented error classification: 400/401/403/404/422/500 with error codes."),
        ("The search feature uses LIKE queries on PostgreSQL.", "Search upgraded to use PostgreSQL full-text search with tsvector indexes."),
    ]

    for i, (old_content, new_content) in enumerate(ku_pairs):
        old_date = (now - timedelta(days=60)).isoformat()
        new_date = (now - timedelta(days=2)).isoformat()
        memories.append({
            "content": old_content,
            "metadata": {
                "event_type": "decision", "priority": 3,
                "session_id": f"bench-ku-old-{i}",
                "referenced_date": old_date,
                "feedback_score": -1,  # Slightly downvoted (outdated)
            },
        })
        memories.append({
            "content": new_content,
            "metadata": {
                "event_type": "decision", "priority": 4,
                "session_id": f"bench-ku-new-{i}",
                "referenced_date": new_date,
                "feedback_score": 2,  # Upvoted
            },
        })

    # Seed all memories
    for mem in memories:
        store.store(
            content=mem["content"],
            session_id=mem["metadata"].get("session_id"),
            metadata=mem["metadata"],
            skip_inference=True,  # Bypass embedding dedup (sprint memories are intentionally similar)
        )

    return len(memories)


def run_benchmark(store, verbose=False):
    """Run the benchmark and return results."""
    results = {
        "information_extraction": {"total": 0, "correct": 0, "details": []},
        "multi_session": {"total": 0, "correct": 0, "details": []},
        "temporal": {"total": 0, "correct": 0, "details": []},
        "knowledge_update": {"total": 0, "correct": 0, "details": []},
        "abstention": {"total": 0, "correct": 0, "details": []},
    }

    def check_top3(query_text, expected_substring, category, **kwargs):
        """Check if any top-3 result contains expected_substring."""
        hits = store.query(query_text, limit=3, **kwargs)
        found = any(expected_substring.lower() in r.content.lower() for r in hits)
        results[category]["total"] += 1
        if found:
            results[category]["correct"] += 1
        if verbose:
            status = "PASS" if found else "FAIL"
            results[category]["details"].append(
                f"  [{status}] Q: {query_text[:60]}  E: {expected_substring[:40]}"
            )
        return found

    # === Information Extraction ===
    check_top3("Python GIL threading", "GIL prevents true parallel", "information_extraction")
    check_top3("SQLite WAL mode concurrent reads", "WAL mode allows concurrent", "information_extraction")
    check_top3("ONNX Runtime memory usage", "337MB RAM", "information_extraction")
    check_top3("React useEffect cleanup", "cleanup function runs", "information_extraction")
    check_top3("Docker layer caching strategy", "rarely-changing instructions", "information_extraction")
    check_top3("PostgreSQL VACUUM schedule", "VACUUM ANALYZE", "information_extraction")
    check_top3("TypeScript discriminated unions", "literal type field", "information_extraction")
    check_top3("Redis KEYS command production", "SCAN is preferred", "information_extraction")
    check_top3("git rebase onto", "rebase --onto", "information_extraction")
    check_top3("Kubernetes liveness readiness probes", "liveness probes should NOT check", "information_extraction")
    check_top3("user preferred CSS framework", "Tailwind CSS", "information_extraction")
    check_top3("SQL injection prevention", "parameterized queries", "information_extraction")
    check_top3("asyncio gather exceptions", "return_exceptions=True", "information_extraction")
    check_top3("nginx proxy_pass trailing slash", "strips the location prefix", "information_extraction")
    check_top3("JWT token expiration best practice", "short-lived", "information_extraction")
    check_top3("user timezone", "America/New_York", "information_extraction")
    check_top3("Rust shared mutable state threads", "Arc<Mutex<T>>", "information_extraction")
    check_top3("Next.js server components client", "use client", "information_extraction")
    check_top3("CoreML memory leak Apple Silicon", "memory leaks", "information_extraction")
    check_top3("state management library choice", "Zustand", "information_extraction")

    # === Multi-session reasoning ===
    check_top3("OMEGA database backend decision", "SQLite for the OMEGA backend", "multi_session")
    check_top3("OMEGA vector search implementation", "sqlite-vec extension", "multi_session")
    check_top3("OMEGA text search", "FTS5 full-text search", "multi_session")
    check_top3("API framework choice", "FastAPI", "multi_session")
    check_top3("rate limiting configuration", "100 req/min", "multi_session")
    check_top3("main database migration from JSON", "PostgreSQL for the main application", "multi_session")
    check_top3("connection pooling solution", "pgbouncer", "multi_session")
    check_top3("retry logic external API", "exponential backoff", "multi_session")
    check_top3("thundering herd prevention", "jitter", "multi_session")
    check_top3("container deployment platform", "ECS with Fargate", "multi_session")
    check_top3("monitoring alerts thresholds", "CPU > 80%", "multi_session")
    check_top3("CI/CD pipeline platform", "GitHub Actions", "multi_session")
    check_top3("database migration in CI/CD", "migration step", "multi_session")
    check_top3("feature flag service", "LaunchDarkly", "multi_session")
    check_top3("deployment risk reduction strategy", "disable new features", "multi_session")
    check_top3("OAuth authentication providers", "Google and GitHub", "multi_session")
    check_top3("PKCE authorization code", "prevent authorization code interception", "multi_session")
    check_top3("metrics collection tool", "Prometheus", "multi_session")
    check_top3("dashboard visualization", "Grafana", "multi_session")
    check_top3("business KPI metrics", "user signups", "multi_session")

    # === Temporal reasoning ===
    now = datetime.now(timezone.utc)
    week_ago = (now - timedelta(days=7)).isoformat()
    check_top3("recent sprint completions", "Sprint 20 completed", "temporal",
               temporal_range=(week_ago, now.isoformat()))
    check_top3("sprint deployments last week", "deployed feature batch", "temporal",
               temporal_range=(week_ago, now.isoformat()))
    two_weeks = (now - timedelta(days=14)).isoformat()
    check_top3("sprint completions last two weeks", "Sprint 1", "temporal",
               temporal_range=(two_weeks, now.isoformat()))
    month_ago = (now - timedelta(days=30)).isoformat()
    check_top3("what was deployed last month", "Sprint", "temporal",
               temporal_range=(month_ago, now.isoformat()))
    # These should NOT match (too old) — reversed expectation
    for i in range(4):
        old_range_start = (now - timedelta(days=90)).isoformat()
        old_range_end = (now - timedelta(days=80)).isoformat()
        hits = store.query("sprint completion", limit=3, temporal_range=(old_range_start, old_range_end))
        results["temporal"]["total"] += 1
        # We expect NO results in this very old range (all sprints are within 60 days)
        if not hits:
            results["temporal"]["correct"] += 1
        elif verbose:
            results["temporal"]["details"].append(
                f"  [FAIL] Expected no results for 80-90 days ago range, got {len(hits)}"
            )
    # Recent range tests
    for days_window in [3, 7, 14, 21, 30, 45, 60]:
        start = (now - timedelta(days=days_window)).isoformat()
        hits = store.query("sprint deployed feature batch", limit=3,
                           temporal_range=(start, now.isoformat()))
        results["temporal"]["total"] += 1
        if hits:
            results["temporal"]["correct"] += 1
        elif verbose:
            results["temporal"]["details"].append(
                f"  [FAIL] Expected results for last {days_window} days, got 0"
            )
    # 5 more temporal queries
    for i in range(5):
        start = (now - timedelta(days=10 + i * 10)).isoformat()
        end = (now - timedelta(days=i * 10)).isoformat()
        hits = store.query("sprint completed production", limit=3,
                           temporal_range=(start, end))
        results["temporal"]["total"] += 1
        if hits:
            results["temporal"]["correct"] += 1

    # === Knowledge update ===
    check_top3("API response format", "JSON for all endpoints", "knowledge_update")
    check_top3("database backup frequency", "every 6 hours", "knowledge_update")
    check_top3("frontend build tooling", "Vite for 10x faster", "knowledge_update")
    check_top3("authentication token expiration", "15 minutes with refresh", "knowledge_update")
    check_top3("application hosting infrastructure", "ECS Fargate with auto-scaling", "knowledge_update")
    check_top3("logging implementation", "structured logging with Winston", "knowledge_update")
    check_top3("test execution workflow", "automatically on every pull request", "knowledge_update")
    check_top3("database schema migration system", "Alembic for database schema", "knowledge_update")
    check_top3("error handling HTTP responses", "error classification", "knowledge_update")
    check_top3("search implementation", "full-text search with tsvector", "knowledge_update")
    # Verify old versions are ranked LOWER
    for query_text, old_substr in [
        ("API response format", "XML for all endpoints"),
        ("database backup frequency", "daily at 2 AM"),
        ("frontend build tooling", "uses Create React App"),
        ("authentication token expiration", "expire after 24 hours"),
        ("application hosting", "single EC2 instance"),
    ]:
        hits = store.query(query_text, limit=3)
        # Old version should NOT be in top-1 (new version should outrank it)
        results["knowledge_update"]["total"] += 1
        if hits and old_substr.lower() not in hits[0].content.lower():
            results["knowledge_update"]["correct"] += 1
        elif verbose:
            top = hits[0].content[:60] if hits else "NO RESULTS"
            results["knowledge_update"]["details"].append(
                f"  [FAIL] Old version ranked #1: {top}"
            )
    # 5 more KU tests
    for query_text, new_substr in [
        ("logging approach", "structured logging"),
        ("test automation", "automatically"),
        ("migration system", "Alembic"),
        ("error responses", "classification"),
        ("search upgrade", "tsvector"),
    ]:
        check_top3(query_text, new_substr, "knowledge_update")

    # === Abstention ===
    # Queries about topics NOT in the memory store — should return low/no results
    irrelevant_queries = [
        "quantum computing superconductor temperature",
        "recipe for chocolate cake ingredients",
        "stock market prediction algorithm",
        "ancient Roman history gladiator battles",
        "knitting patterns for winter sweaters",
        "deep sea marine biology bioluminescence",
        "amateur radio frequency bands",
        "origami crane folding instructions",
        "volcanic eruption prediction methods",
        "medieval castle architecture design",
        "astrophotography camera settings Milky Way",
        "woodworking dovetail joint techniques",
        "cheese aging cave temperature humidity",
        "hot air balloon flight physics",
        "crossword puzzle solving strategies",
        "beekeeping hive inspection schedule",
        "surfboard shaping foam blank",
        "calligraphy brush stroke techniques",
        "gem cutting faceting angles",
        "sourdough bread starter maintenance",
    ]
    for q in irrelevant_queries:
        hits = store.query(q, limit=3)
        results["abstention"]["total"] += 1
        # For abstention: "correct" means low relevance (< 0.3) or no results
        if not hits or all(r.relevance < 0.3 for r in hits):
            results["abstention"]["correct"] += 1
        elif verbose:
            top_rel = hits[0].relevance if hits else 0
            results["abstention"]["details"].append(
                f"  [FAIL] Q: {q[:40]}  top_relevance={top_rel:.2f}"
            )

    return results


def print_results(results):
    """Print formatted benchmark results."""
    print("\n" + "=" * 60)
    print("  OMEGA LongMemEval Benchmark Results")
    print("=" * 60)

    total_correct = 0
    total_questions = 0
    categories = [
        ("information_extraction", "Information Extraction"),
        ("multi_session", "Multi-Session Reasoning"),
        ("temporal", "Temporal Reasoning"),
        ("knowledge_update", "Knowledge Update"),
        ("abstention", "Abstention"),
    ]

    for key, label in categories:
        cat = results[key]
        correct = cat["correct"]
        total = cat["total"]
        pct = (correct / total * 100) if total > 0 else 0
        bar = "#" * int(pct / 5) + "-" * (20 - int(pct / 5))
        grade = "A" if pct >= 90 else "B" if pct >= 75 else "C" if pct >= 60 else "D" if pct >= 40 else "F"
        print(f"\n  {label:30s} {correct:3d}/{total:3d}  [{bar}] {pct:5.1f}%  ({grade})")

        total_correct += correct
        total_questions += total

        for detail in cat.get("details", []):
            print(detail)

    overall = (total_correct / total_questions * 100) if total_questions > 0 else 0
    print(f"\n{'─' * 60}")
    print(f"  OVERALL: {total_correct}/{total_questions} = {overall:.1f}%")
    print(f"{'─' * 60}\n")

    return overall


def main():
    parser = argparse.ArgumentParser(description="OMEGA LongMemEval Benchmark")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show per-question details")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = str(Path(tmpdir) / "bench.db")
        print("Creating benchmark database...")
        store = create_bench_store(db_path)

        print("Seeding memories...")
        count = seed_memories(store)
        print(f"Seeded {count} memories.")

        print("Running benchmark...")
        results = run_benchmark(store, verbose=args.verbose)
        overall = print_results(results)

        store.close()

    return 0 if overall >= 50 else 1


if __name__ == "__main__":
    sys.exit(main())
