"""
OMEGA Bridge -- High-level API for OMEGA memory system.

Provides the public interface used by the MCP server handlers.
All functions are thin wrappers that delegate to the SQLiteStore singleton.

Public API (36 functions, see __all__ for full list):
    Core:       auto_capture, store, remember, delete_memory, edit_memory
    Query:      query, query_structured, phrase_search, find_similar_memories
    Session:    welcome, clear_session, batch_store
    Health:     check_health, status, get_dedup_stats
    Profile:    get_profile, save_profile, extract_preferences, list_preferences
    Lessons:    get_cross_session_lessons, get_cross_project_lessons
    Maintenance: consolidate, compact, deduplicate, timeline, traverse
    Export:     export_memories, import_memories, reingest
    Stats:      type_stats, session_stats
    Constraints: check_constraints, list_constraints, save_constraints
    Feedback:   record_feedback
    Testing:    reset_memory
"""

import atexit
import logging
import os
import re
import threading
import unicodedata
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from omega import json_compat as json
from omega.types import TTLCategory, AutoCaptureEventType

logger = logging.getLogger("omega.bridge")


# ---------------------------------------------------------------------------
# Storage configuration
# ---------------------------------------------------------------------------

OMEGA_HOME = Path(os.environ.get("OMEGA_HOME", str(Path.home() / ".omega")))

# Per-event-type dedup thresholds for Jaccard similarity.
DEDUP_THRESHOLDS: Dict[str, float] = {
    AutoCaptureEventType.ERROR_PATTERN: 0.70,
    AutoCaptureEventType.SESSION_SUMMARY: 0.75,
    AutoCaptureEventType.TASK_COMPLETION: 0.85,
    AutoCaptureEventType.DECISION: 0.80,
    AutoCaptureEventType.LESSON_LEARNED: 0.85,
    AutoCaptureEventType.CHECKPOINT: 0.90,
}

# Event types that participate in memory evolution (Zettelkasten-style).
EVOLUTION_TYPES = {
    AutoCaptureEventType.LESSON_LEARNED,
    AutoCaptureEventType.DECISION,
    AutoCaptureEventType.ERROR_PATTERN,
}
EVOLUTION_THRESHOLD = 0.65

# Content blocklist — reject system noise at ingestion time.
# Startswith patterns (checked against content[:50])
_BLOCKLIST_STARTSWITH = [
    "[BROADCAST",
    "[WORK BREADCRUMB",
    "[WORK DISPATCH",
    "<task-notification>",
    "Decision: <task-notification>",
]
# Substring patterns (checked anywhere in content)
_BLOCKLIST_CONTAINS = [
    '"error":',
    '"stderr":',
    '"stdout":',
    "[BROADCAST",
]

# Minimum content length for auto-capture (reject very short noise).
# Raised from 40 to 80 to filter infrastructure noise that inflates never-accessed count.
_MIN_CONTENT_LENGTH = 80

# Event types from hooks that generate infrastructure noise (never accessed, inflate DB).
_INFRASTRUCTURE_EVENT_TYPES = frozenset({
    "consolidate", "compact", "checkpoint", "coordination_snapshot",
    "session_respawn", "file_summary", "code_chunk",
})


def _check_milestone(name: str) -> bool:
    """Return True if milestone not yet achieved (first time). Creates marker."""
    marker = OMEGA_HOME / "milestones" / name
    if marker.exists():
        return False
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.touch()
    return True


# ---------------------------------------------------------------------------
# Lazy singleton -- SQLiteStore replaces OmegaMemory
# ---------------------------------------------------------------------------

_store_instance = None
_store_lock = threading.Lock()


def _get_store():
    """Get or create the SQLiteStore singleton (thread-safe).

    Uses local variable for init to ensure _store_instance is only set
    after full initialization (migration + cleanup + atexit registration).
    """
    global _store_instance
    if _store_instance is not None:
        return _store_instance
    with _store_lock:
        if _store_instance is not None:
            return _store_instance
        # Auto-migrate from JSON graphs if needed (first run after upgrade)
        from omega.migrate_to_sqlite import auto_migrate_if_needed

        auto_migrate_if_needed()

        from omega.sqlite_store import SQLiteStore

        # Init into local var first; only publish to global after full setup
        store = SQLiteStore()
        # Purge expired nodes on startup
        expired = store.cleanup_expired()
        if expired > 0:
            logger.info(f"Startup: purged {expired} expired nodes")
        atexit.register(_close_store)
        _store_instance = store  # Publish only after full init
    return _store_instance


def _close_store():
    """Close SQLiteStore on process exit."""
    global _store_instance
    if _store_instance is not None:
        try:
            _store_instance.close()
        except Exception as e:
            logger.debug("Store close failed during refresh: %s", e)


def reset_memory():
    """Reset the singleton (useful for testing)."""
    global _store_instance
    if _store_instance is not None:
        try:
            _store_instance.close()
        except Exception as e:
            logger.debug("Store close failed during reset: %s", e)
    _store_instance = None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _human_ttl(ttl: Optional[int]) -> str:
    """Format TTL seconds as human-readable string."""
    if not ttl:
        return "permanent"
    if ttl < 3600:
        return f"{ttl // 60}m"
    if ttl < 86400:
        return f"{ttl // 3600}h"
    return f"{ttl // 86400}d"


def _normalize_for_dedup(text: str) -> str:
    """Normalize text for dedup comparison by stripping variable parts."""
    t = text.lower()
    t = re.sub(r"[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}", "<ID>", t)
    t = re.sub(r"/[\w/.\-]+\.\w{1,5}", "<PATH>", t)
    t = re.sub(r"'[^']{1,80}'", "<NAME>", t)
    t = re.sub(r'"[^"]{1,80}"', "<NAME>", t)
    t = re.sub(r"\b\d+\b", "<N>", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


_TAG_LANGUAGES = {
    "python",
    "javascript",
    "typescript",
    "rust",
    "go",
    "java",
    "ruby",
    "swift",
    "kotlin",
    "c++",
    "c#",
    "php",
    "sql",
    "bash",
    "zsh",
    "html",
    "css",
}
_TAG_TOOLS = {
    "react",
    "next.js",
    "nextjs",
    "django",
    "flask",
    "fastapi",
    "docker",
    "kubernetes",
    "git",
    "npm",
    "pip",
    "pytest",
    "webpack",
    "vite",
    "redis",
    "postgres",
    "sqlite",
    "mongodb",
    "aws",
    "gcp",
    "azure",
    "vercel",
    "nginx",
    "mysql",
    "jest",
    "vitest",
    "yarn",
    "pnpm",
    "bun",
    "deno",
    "supabase",
    "onnx",
    "mcp",
    "asyncio",
    "threading",
    "sqlalchemy",
    "celery",
    "graphql",
    "prisma",
    "terraform",
    "ansible",
    "helm",
    "zustand",
    "tailwind",
    "shadcn",
    "storybook",
    "playwright",
    "cypress",
    "openai",
    "anthropic",
    "langchain",
    "chromadb",
    "pinecone",
    "homebrew",
    "launchd",
    "systemd",
    "cron",
}
_TAG_ALIASES = {
    "postgresql": "postgres",
    "k8s": "kubernetes",
    "js": "javascript",
    "ts": "typescript",
    "py": "python",
    "rb": "ruby",
    "tf": "terraform",
    "cdk": "aws-cdk",
    "nextjs": "next.js",
    "reactjs": "react",
    "sqlite3": "sqlite",
    "onnxruntime": "onnx",
    "fts5": "sqlite",
    "sqlitevec": "sqlite",
}
_GO_CONTEXT_WORDS = {"goroutine", "goroutines", "chan", "defer", "func", "gomod", "gofmt"}


_TAG_CONCEPTS = {
    "hook",
    "hooks",
    "daemon",
    "migration",
    "api",
    "config",
    "configuration",
    "testing",
    "test",
    "tests",
    "debug",
    "debugging",
    "performance",
    "cache",
    "caching",
    "auth",
    "authentication",
    "authorization",
    "deploy",
    "deployment",
    "refactor",
    "refactoring",
    "schema",
    "embedding",
    "embeddings",
    "vector",
    "coordination",
    "concurrency",
    "async",
    "sync",
}

# Map concept words to canonical tags
_CONCEPT_CANONICAL = {
    "hooks": "hook",
    "tests": "testing",
    "test": "testing",
    "debugging": "debug",
    "caching": "cache",
    "authentication": "auth",
    "authorization": "auth",
    "deployment": "deploy",
    "refactoring": "refactor",
    "configuration": "config",
    "embeddings": "embedding",
}


def _extract_tags(content: str, project: Optional[str] = None) -> List[str]:
    """Extract auto-tags from content (languages, tools, file paths, concepts, project)."""
    tags: set = set()
    words = set(re.findall(r"\b[\w.+#]+\b", content.lower()))
    # Apply aliases first (e.g. "postgresql" -> "postgres")
    for alias, canonical in _TAG_ALIASES.items():
        if alias in words:
            tags.add(canonical)
    # Languages (with Go disambiguation)
    for w in words:
        if w in _TAG_LANGUAGES:
            if w == "go":
                # Only tag "go" if Go-specific context words are present
                if words & _GO_CONTEXT_WORDS:
                    tags.add(w)
            else:
                tags.add(w)
    tags.update(w for w in words if w in _TAG_TOOLS)
    # Concepts (hook, testing, auth, etc.)
    for w in words:
        if w in _TAG_CONCEPTS:
            tags.add(_CONCEPT_CANONICAL.get(w, w))
    # File paths
    for match in re.findall(r"(?:/[\w.\-]+){2,}", content):
        tags.add(match)
    # File extensions mentioned inline (e.g. ".py", ".ts")
    for ext in re.findall(r"\b\w+\.(py|js|ts|tsx|rs|go|rb|java|swift|sql|sh|yaml|json|toml)\b", content.lower()):
        ext_map = {
            "py": "python",
            "js": "javascript",
            "ts": "typescript",
            "tsx": "typescript",
            "rs": "rust",
            "rb": "ruby",
            "sh": "bash",
        }
        if ext in ext_map:
            tags.add(ext_map[ext])
    # Project name
    if project:
        tags.add(Path(project).name.lower())
    return sorted(tags)[:10]


def _infer_temporal_range(query_text: str) -> Optional[tuple]:
    """Infer a (start_iso, end_iso) temporal range from natural-language time references.

    Supports: "last week", "yesterday", "N days/hours ago", "today",
    "this week/month/year", month names, day-of-week references,
    "the week/month of <date>", ISO dates.
    Returns None if no temporal signal is detected.
    """
    from datetime import timedelta

    now = datetime.now(timezone.utc)
    text = query_text.lower()

    # "last N days/hours/weeks/months/years"
    m = re.search(r"last\s+(\d+)\s+(day|hour|week|month|year)s?", text)
    if m:
        n, unit = int(m.group(1)), m.group(2)
        delta = {
            "day": timedelta(days=n),
            "hour": timedelta(hours=n),
            "week": timedelta(weeks=n),
            "month": timedelta(days=n * 30),
            "year": timedelta(days=n * 365),
        }[unit]
        return ((now - delta).isoformat(), now.isoformat())

    # "N days/hours ago"
    m = re.search(r"(\d+)\s+(day|hour|week|month|year)s?\s+ago", text)
    if m:
        n, unit = int(m.group(1)), m.group(2)
        delta = {
            "day": timedelta(days=n),
            "hour": timedelta(hours=n),
            "week": timedelta(weeks=n),
            "month": timedelta(days=n * 30),
            "year": timedelta(days=n * 365),
        }[unit]
        return ((now - delta).isoformat(), now.isoformat())

    if "yesterday" in text:
        start = (now - timedelta(days=1)).replace(hour=0, minute=0, second=0)
        end = start + timedelta(days=1)
        return (start.isoformat(), end.isoformat())

    if "today" in text:
        start = now.replace(hour=0, minute=0, second=0)
        return (start.isoformat(), now.isoformat())

    if "last week" in text:
        # Previous Mon-Sun week
        days_since_monday = now.weekday()
        last_monday = now - timedelta(days=days_since_monday + 7)
        start = last_monday.replace(hour=0, minute=0, second=0)
        end = start + timedelta(days=7)
        return (start.isoformat(), end.isoformat())

    if "this week" in text:
        days_since_monday = now.weekday()
        start = (now - timedelta(days=days_since_monday)).replace(hour=0, minute=0, second=0)
        return (start.isoformat(), now.isoformat())

    if "last month" in text:
        # Previous calendar month
        first_this_month = now.replace(day=1, hour=0, minute=0, second=0)
        end = first_this_month
        if now.month == 1:
            start = datetime(now.year - 1, 12, 1, tzinfo=timezone.utc)
        else:
            start = datetime(now.year, now.month - 1, 1, tzinfo=timezone.utc)
        return (start.isoformat(), end.isoformat())

    if "this month" in text:
        start = now.replace(day=1, hour=0, minute=0, second=0)
        return (start.isoformat(), now.isoformat())

    if "this year" in text:
        start = datetime(now.year, 1, 1, tzinfo=timezone.utc)
        return (start.isoformat(), now.isoformat())

    if "last year" in text:
        start = datetime(now.year - 1, 1, 1, tzinfo=timezone.utc)
        end = datetime(now.year, 1, 1, tzinfo=timezone.utc)
        return (start.isoformat(), end.isoformat())

    # Day-of-week references: "last Monday", "on Friday", etc.
    _DAYS = {"monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
             "friday": 4, "saturday": 5, "sunday": 6}
    for day_name, day_num in _DAYS.items():
        if day_name in text:
            # Find the most recent occurrence of this day
            days_ago = (now.weekday() - day_num) % 7
            if days_ago == 0:
                days_ago = 7  # "last Monday" means previous, not today
            target = now - timedelta(days=days_ago)
            start = target.replace(hour=0, minute=0, second=0)
            end = start + timedelta(days=1)
            return (start.isoformat(), end.isoformat())

    # "Month YYYY" or "in Month YYYY" (e.g., "January 2025")
    months = {
        "january": 1, "february": 2, "march": 3, "april": 4,
        "may": 5, "june": 6, "july": 7, "august": 8,
        "september": 9, "october": 10, "november": 11, "december": 12,
    }
    for name, num in months.items():
        # Check for "Month YYYY" first
        m = re.search(rf"\b{name}\s+(\d{{4}})\b", text)
        if m:
            year = int(m.group(1))
            start = datetime(year, num, 1, tzinfo=timezone.utc)
            if num == 12:
                end = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
            else:
                end = datetime(year, num + 1, 1, tzinfo=timezone.utc)
            return (start.isoformat(), end.isoformat())
        # Bare month name (assume most recent occurrence)
        if name in text:
            year = now.year if num <= now.month else now.year - 1
            start = datetime(year, num, 1, tzinfo=timezone.utc)
            if num == 12:
                end = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
            else:
                end = datetime(year, num + 1, 1, tzinfo=timezone.utc)
            return (start.isoformat(), end.isoformat())

    # ISO date (YYYY-MM-DD)
    m = re.search(r"(\d{4}-\d{2}-\d{2})", text)
    if m:
        date_str = m.group(1)
        start = datetime.fromisoformat(date_str).replace(tzinfo=timezone.utc)
        end = start + timedelta(days=1)
        return (start.isoformat(), end.isoformat())

    # Bare year reference (e.g., "in 2024")
    m = re.search(r"\bin\s+(20\d{2})\b", text)
    if m:
        year = int(m.group(1))
        start = datetime(year, 1, 1, tzinfo=timezone.utc)
        end = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
        return (start.isoformat(), end.isoformat())

    return None


def _relative_time(created_at) -> str:
    """Format a datetime as a human-readable relative time string."""
    if not created_at:
        return ""
    now = datetime.now(timezone.utc)
    if isinstance(created_at, str):
        try:
            if created_at.endswith("Z"):
                created_at = created_at[:-1] + "+00:00"
            created_at = datetime.fromisoformat(created_at)
        except (ValueError, TypeError):
            return ""
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=timezone.utc)
    delta = now - created_at
    seconds = delta.total_seconds()
    if seconds < 60:
        return "just now"
    if seconds < 3600:
        m = int(seconds // 60)
        return f"{m}m ago"
    if seconds < 86400:
        h = int(seconds // 3600)
        return f"{h}h ago"
    days = int(seconds // 86400)
    if days == 1:
        return "yesterday"
    if days < 30:
        return f"{days}d ago"
    months = days // 30
    if months == 1:
        return "1 month ago"
    return f"{months} months ago"


def _compress_to_observation(content: str, event_type: str = "") -> Optional[str]:
    """Compress content to a concise observation (extractive, no LLM).

    Selects the 1-2 most information-dense sentences from the content.
    Returns None if content is already concise (< 150 chars) or compression fails.
    """
    if len(content) < 150:
        return None  # Already concise

    # Split into sentences (preserve abbreviations, version numbers)
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z\[])", content)
    if len(sentences) <= 1:
        # Try simpler split
        sentences = re.split(r"(?<=[.!?])\s+", content)
    sentences = [s.strip() for s in sentences if len(s.strip()) >= 15]

    if not sentences:
        return None

    # Score each sentence by information density
    scored = []
    for s in sentences:
        words = s.split()
        unique_words = len(set(w.lower() for w in words if len(w) > 3))
        # Bonus for code tokens (backticks, paths, CamelCase)
        code_tokens = len(re.findall(r"`[^`]+`|/[\w/.]+|\b[A-Z][a-z]+[A-Z]\w*\b", s))
        # Diminishing returns on length
        length_score = min(len(s), 200) / 200.0
        density = unique_words * 1.0 + code_tokens * 2.0 + length_score * 3.0
        scored.append((density, s))

    scored.sort(key=lambda x: x[0], reverse=True)

    # Select top 1-2 diverse sentences
    selected = [scored[0][1]]
    if len(scored) > 1:
        # Add second only if sufficiently different
        s2 = scored[1][1]
        if _jaccard(selected[0].lower(), s2.lower(), min_word_len=3) < 0.7:
            selected.append(s2)

    observation = " ".join(selected)
    if len(observation) > 200:
        observation = observation[:197] + "..."

    return observation


def _jaccard(text_a: str, text_b: str, min_word_len: int = 4) -> float:
    """Jaccard similarity on word sets (fast, no embeddings)."""
    words_a = {w for w in text_a.split() if len(w) >= min_word_len}
    words_b = {w for w in text_b.split() if len(w) >= min_word_len}
    if not words_a or not words_b:
        return 0.0
    return len(words_a & words_b) / len(words_a | words_b)


def _auto_relate(store, node_id: str, max_related: int = 3, min_similarity: float = 0.45) -> int:
    """Create 'related' edges from node_id to its most similar existing memories.

    Returns the number of edges created. Silently returns 0 on any error.
    """
    try:
        embedding = store.get_embedding(node_id)
        if not embedding:
            return 0
        similar = store.find_similar(embedding, limit=max_related + 1)
        related = [r for r in similar if r.id != node_id and r.relevance >= min_similarity][:max_related]
        count = 0
        for r in related:
            if store.add_edge(node_id, r.id, "related", r.relevance):
                count += 1
        if count:
            logger.debug(f"Auto-related {node_id[:12]} to {count} memories")
        return count
    except Exception as e:
        logger.debug(f"_auto_relate failed for {node_id[:12]}: {e}")
        return 0


# ---------------------------------------------------------------------------
# Public API -- Core CRUD
# ---------------------------------------------------------------------------


def auto_capture(
    content: str,
    event_type: str,
    metadata: Optional[Dict[str, Any]] = None,
    session_id: Optional[str] = None,
    project: Optional[str] = None,
    ttl_override: Optional[int] = None,
    entity_id: Optional[str] = None,
    agent_type: Optional[str] = None,
) -> str:
    """Store a memory with auto-classification, dedup, and evolution.

    This is the primary ingestion function. It:
    1. Checks for near-duplicate content (Jaccard) and reuses if found.
    2. Tries to *evolve* an existing memory with new insights (Zettelkasten).
    3. Falls back to creating a new memory node.

    Returns:
        Markdown confirmation string.
    """
    content = unicodedata.normalize("NFC", content)

    # Auto-resolve entity_id from project if not explicitly provided
    if not entity_id and project:
        try:
            from omega.entity.engine import resolve_project_entity
            entity_id = resolve_project_entity(project)
        except Exception as e:
            logger.debug("Entity resolution failed: %s", e)

    # Determine source early — hooks vs direct API calls have different filtering rules.
    _source = (metadata or {}).get("source", "")
    _is_hook = _source.startswith("auto_") or _source.endswith("_hook")

    # Block system noise early — startswith patterns are position-specific, safe for all sources.
    for pattern in _BLOCKLIST_STARTSWITH:
        if content.startswith(pattern):
            return "**Memory Blocked** (system noise)"
    # Contains patterns only apply to hook-sourced content to avoid false positives
    # on direct API calls (e.g. storing a lesson that mentions "error":).
    if _is_hook:
        for pattern in _BLOCKLIST_CONTAINS:
            if pattern in content:
                return "**Memory Blocked** (system noise)"

    # Min-length gate — only for auto-captured content from hooks, not direct API calls.
    if _is_hook and len(content) < _MIN_CONTENT_LENGTH and event_type != AutoCaptureEventType.USER_PREFERENCE:
        return "**Memory Blocked** (too short)"

    # Block infrastructure event types that generate noise and inflate never-accessed count
    if _is_hook and event_type in _INFRASTRUCTURE_EVENT_TYPES:
        return "**Memory Blocked** (infrastructure noise)"

    # Block zero-value outcome records (tokens=0 partial sessions)
    if _is_hook and event_type == "task_completion" and "tokens=0" in content:
        return "**Memory Blocked** (zero-token outcome)"

    # Block JSON-blob decisions — raw tool output stored as "decisions"
    if event_type == "decision":
        # Strip known prefixes to check the actual body
        _body = content
        for _pfx in ("Decision: ", "Plan/decision captured: ", "Fact: "):
            if _body.startswith(_pfx):
                _body = _body[len(_pfx):]
        _body_stripped = _body.lstrip()
        if _body_stripped.startswith(("{", "[", '"filePath', '"type"')):
            return "**Memory Blocked** (JSON blob, not a decision)"

    store = _get_store()
    meta = dict(metadata or {})
    meta["event_type"] = event_type
    if project:
        meta["project"] = project
    meta["captured_at"] = datetime.now(timezone.utc).isoformat()

    # Set capture confidence (if not already set by caller)
    if not meta.get("capture_confidence"):
        source = meta.get("source", "")
        if source == "user_remember":
            meta["capture_confidence"] = "high"
        elif event_type == "user_preference":
            meta["capture_confidence"] = "high"
        elif event_type in ("lesson_learned", "error_pattern") and not source.startswith("auto_"):
            # Direct API calls for lessons/errors = validated by agent
            meta["capture_confidence"] = "high"
        elif source in ("auto_plan_capture",):
            # Auto-captured plans are speculative
            meta["capture_confidence"] = "low"
        elif source.startswith("auto_") or source.endswith("_hook"):
            meta["capture_confidence"] = "medium"
        else:
            # Direct API calls (agent-initiated store) = higher trust
            meta["capture_confidence"] = "high"

    # Auto-tag extraction
    auto_tags = _extract_tags(content, project)
    if auto_tags:
        existing_tags = meta.get("tags", [])
        meta["tags"] = sorted(set(existing_tags + auto_tags))[:15]

    ttl = ttl_override if ttl_override is not None else TTLCategory.for_event_type(event_type)

    # ------------------------------------------------------------------
    # Phase 1 + 2: Content dedup, error burst, and evolution
    # ------------------------------------------------------------------
    # Single query for both dedup and evolution (same search text).
    # This avoids duplicate embedding generation + DB round-trips.
    dedup_threshold = DEDUP_THRESHOLDS.get(event_type)
    _similar_results = None  # Lazy-loaded, shared between dedup and evolution

    if dedup_threshold is not None or event_type in EVOLUTION_TYPES:
        try:
            _similar_results = store.query(content[:200], limit=8)
        except Exception as e:
            logger.debug(f"Similar-content query failed: {e}")

    # Phase 1: Content-level dedup
    if dedup_threshold is not None and _similar_results:
        try:
            for existing in _similar_results:
                if (existing.metadata or {}).get("event_type", "") != event_type:
                    continue
                # Session filter for dedup: only dedup within same session
                # Exception: decisions dedup cross-session (same architectural choice restated)
                if session_id and event_type != AutoCaptureEventType.DECISION:
                    existing_session = (existing.metadata or {}).get("session_id", "")
                    if existing_session and existing_session != session_id:
                        continue
                if event_type == AutoCaptureEventType.ERROR_PATTERN:
                    sim = _jaccard(_normalize_for_dedup(content), _normalize_for_dedup(existing.content))
                else:
                    sim = _jaccard(content.lower(), existing.content.lower())
                if sim > dedup_threshold:
                    store.update_node(existing.id, access_count=(existing.access_count or 0) + 1)
                    store.stats.setdefault("content_dedup_skips", 0)
                    store.stats["content_dedup_skips"] += 1
                    linked = _auto_relate(store, existing.id)
                    logger.debug(f"Content dedup: skipped {event_type} (jaccard={sim:.2f}), reusing {existing.id[:12]}")
                    link_info = f", {linked} linked" if linked else ""
                    return f"Deduped → {existing.id[:12]}{link_info}"
        except Exception as e:
            logger.debug(f"Content dedup check skipped: {e}")

    # Phase 1.5: Error burst detection
    if event_type == AutoCaptureEventType.ERROR_PATTERN and session_id:
        try:
            # Use similar results if available, otherwise minimal query
            burst_candidates = _similar_results or []
            session_errors = [
                r
                for r in burst_candidates
                if (r.metadata or {}).get("event_type") == AutoCaptureEventType.ERROR_PATTERN
                and (r.metadata or {}).get("session_id") == session_id
            ]
            if len(session_errors) >= 3:
                # Only capture if truly novel (Jaccard < 0.40 with all recent errors)
                is_novel = all(_jaccard(content.lower(), e.content.lower()) < 0.40 for e in session_errors)
                if not is_novel:
                    store.stats.setdefault("error_burst_skips", 0)
                    store.stats["error_burst_skips"] += 1
                    return "Blocked (error burst — duplicate)"
        except Exception as e:
            logger.debug(f"Error burst check skipped: {e}")

    # Phase 2: Memory evolution (Zettelkasten-inspired)
    if event_type in EVOLUTION_TYPES and _similar_results:
        try:
            for existing in _similar_results[:3]:
                if (existing.metadata or {}).get("event_type", "") != event_type:
                    continue
                sim = _jaccard(content.lower(), existing.content.lower())
                if EVOLUTION_THRESHOLD <= sim < (dedup_threshold or 0.95):
                    old_words = {w.lower() for w in existing.content.split() if len(w) > 3}
                    new_info = {w.lower() for w in content.split() if len(w) > 3} - old_words
                    if len(new_info) < 3:
                        continue

                    evolved = existing.content.rstrip()
                    if not evolved.endswith("."):
                        evolved += "."

                    new_sentences = []
                    # Split on sentence boundaries, preserving abbreviations
                    # like "Dr.", "e.g.", "i.e.", version numbers "v2.0"
                    for sentence in re.split(r"(?<=[.!?])\s+(?=[A-Z])", content):
                        sentence = sentence.strip()
                        if not sentence or len(sentence) < 10:
                            continue
                        s_words = {w.lower() for w in sentence.split() if len(w) > 3}
                        if s_words and len(s_words - old_words) >= 2:
                            new_sentences.append(sentence)

                    if new_sentences:
                        addition = " ".join(new_sentences[:2])
                        new_content = f"{evolved} [Updated] {addition}"
                        emeta = dict(existing.metadata or {})
                        evo_count = emeta.get("evolution_count", 0) + 1
                        emeta["evolution_count"] = evo_count
                        emeta["last_evolved"] = datetime.now(timezone.utc).isoformat()
                        emeta["evolved_from_sessions"] = list(
                            set(emeta.get("evolved_from_sessions", []) + ([session_id] if session_id else []))
                        )[:10]

                        store.update_node(
                            existing.id,
                            content=new_content,
                            metadata=emeta,
                            access_count=(existing.access_count or 0) + 1,
                        )

                        store.stats.setdefault("memory_evolutions", 0)
                        store.stats["memory_evolutions"] += 1
                        linked = _auto_relate(store, existing.id)
                        logger.info(f"Memory evolved: {existing.id[:12]} (evolution #{evo_count}, jaccard={sim:.2f})")
                        link_info = f", {linked} linked" if linked else ""
                        return f"Evolved {existing.id[:12]} (#{evo_count}{link_info})"
                    break  # Only try the top match
        except Exception as e:
            logger.debug(f"Memory evolution check skipped: {e}")

    # ------------------------------------------------------------------
    # Phase 3: Store new node
    # ------------------------------------------------------------------
    # Wire entity_id into metadata for tag-based discovery
    if entity_id:
        meta["entity_id"] = entity_id

    # Wire agent_type into metadata for discovery
    if agent_type:
        meta["agent_type"] = agent_type

    node_id = store.store(
        content=content,
        session_id=session_id,
        metadata=meta,
        ttl_seconds=ttl,
        entity_id=entity_id,
        agent_type=agent_type,
    )

    ttl_str = _human_ttl(ttl)
    output = "# Memory Captured\n\n"
    output += f"**Event Type:** {event_type}\n"
    output += f"**Node ID:** `{node_id}`\n"
    output += f"**TTL:** {ttl_str}\n"
    if session_id:
        output += f"**Session:** `{session_id[:20]}...`\n"
    if project:
        output += f"**Project:** {project}\n"
    if entity_id:
        output += f"**Entity:** `{entity_id}`\n"
    if agent_type:
        output += f"**Agent Type:** `{agent_type}`\n"

    # ------------------------------------------------------------------
    # Phase 3.5: Observation compression for high-value types
    # ------------------------------------------------------------------
    _HIGH_VALUE_OBSERVATION_TYPES = {"decision", "lesson_learned", "error_pattern", "user_preference"}
    if event_type in _HIGH_VALUE_OBSERVATION_TYPES:
        try:
            observation = _compress_to_observation(content, event_type)
            if observation:
                meta["observation"] = observation
                store.update_node(node_id, metadata=meta)
        except Exception as e:
            logger.debug(f"Observation compression failed for {node_id[:12]}: {e}")

    # ------------------------------------------------------------------
    # Phase 4: Auto-relate — link to similar existing memories
    # ------------------------------------------------------------------
    linked = _auto_relate(store, node_id)
    if linked:
        output += f"**Related:** {linked} linked\n"

    # ------------------------------------------------------------------
    # Phase 5: Milestone celebrations
    # ------------------------------------------------------------------
    try:
        if _check_milestone("first-capture"):
            output += "\n[OMEGA] First memory captured! Your knowledge base is now growing.\n"
        else:
            # Check for 10th capture milestone
            total = store.node_count()
            if total == 10 and _check_milestone("tenth-capture"):
                output += "\n[OMEGA] 10 memories stored — run omega_weekly_digest to see what OMEGA has learned.\n"
    except Exception:
        pass

    logger.info(f"Auto-captured {event_type}: {node_id}")
    return output


def store(
    content: str,
    event_type: str = "memory",
    metadata: Optional[Dict[str, Any]] = None,
    session_id: Optional[str] = None,
    project: Optional[str] = None,
    entity_id: Optional[str] = None,
    agent_type: Optional[str] = None,
) -> str:
    """Direct store -- wraps auto_capture with a default event type."""
    return auto_capture(
        content=content,
        event_type=event_type,
        metadata=metadata,
        session_id=session_id,
        project=project,
        entity_id=entity_id,
        agent_type=agent_type,
    )


def remember(text: str, session_id: Optional[str] = None) -> str:
    """User-facing 'remember this' -- stores with user_preference type."""
    return auto_capture(
        content=text,
        event_type=AutoCaptureEventType.USER_PREFERENCE,
        session_id=session_id,
        metadata={"source": "user_remember"},
    )


def delete_memory(memory_id: str) -> Dict[str, Any]:
    """Delete a memory by its node ID."""
    db = _get_store()
    try:
        success = db.delete_node(memory_id)
        if success:
            logger.info(f"Deleted memory {memory_id[:12]}")
            return {"success": True, "deleted_id": memory_id}
        return {"success": False, "error": f"Memory {memory_id} not found"}
    except Exception as e:
        logger.error(f"Failed to delete memory {memory_id[:12]}: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


def edit_memory(memory_id: str, new_content: str) -> Dict[str, Any]:
    """Edit a memory's content by its node ID."""
    db = _get_store()
    try:
        node = db.get_node(memory_id)
        if node is None:
            return {"success": False, "error": f"Memory {memory_id} not found"}

        old_preview = node.content[:80]
        emeta = dict(node.metadata or {})
        emeta["edited_at"] = datetime.now(timezone.utc).isoformat()
        emeta["edit_count"] = emeta.get("edit_count", 0) + 1

        db.update_node(memory_id, content=new_content, metadata=emeta)

        logger.info(f"Edited memory {memory_id[:12]}")
        return {
            "success": True,
            "id": memory_id,
            "old_content_preview": old_preview,
            "new_content_preview": new_content[:80],
        }
    except Exception as e:
        logger.error(f"Failed to edit memory {memory_id[:12]}: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


# ---------------------------------------------------------------------------
# Public API -- Query
# ---------------------------------------------------------------------------


def query(
    query_text: str,
    limit: int = 10,
    session_id: Optional[str] = None,
    project: Optional[str] = None,
    event_type: Optional[str] = None,
    context_file: Optional[str] = None,
    context_tags: Optional[List[str]] = None,
    filter_tags: Optional[List[str]] = None,
    temporal_range: Optional[tuple] = None,
    entity_id: Optional[str] = None,
    agent_type: Optional[str] = None,
) -> str:
    """Search memories with optional intent-aware routing.

    Args:
        context_file: Current file being edited (for contextual re-ranking).
        context_tags: Current context tags like language, tools (for re-ranking boost).
        filter_tags: Hard filter — only return memories containing ALL specified tags.
        temporal_range: Optional (start_iso, end_iso) tuple. Auto-inferred from query if not given.

    Returns:
        Formatted markdown string with results.
    """
    db = _get_store()
    query_text = unicodedata.normalize("NFC", query_text)

    try:
        # Auto-infer temporal range from query text if not explicitly provided
        effective_temporal = temporal_range or _infer_temporal_range(query_text)

        enhanced = query_text
        if event_type:
            enhanced = f"{event_type} {enhanced}"
        if project:
            enhanced = f"{Path(project).name} {enhanced}"

        results = db.query(
            enhanced,
            limit=limit * 2 if (filter_tags or entity_id or agent_type) else limit,
            session_id=session_id,
            context_file=context_file or "",
            context_tags=context_tags,
            temporal_range=effective_temporal,
            entity_id=entity_id,
            agent_type=agent_type,
            query_hint=event_type,
        )

        # Filter by event_type if specified
        if event_type and results:
            results = [r for r in results if (r.metadata or {}).get("event_type") == event_type]

        # Hard filter by tags (AND logic — all specified tags must be present)
        if filter_tags and results:
            filter_set = {t.lower() for t in filter_tags}
            results = [
                r for r in results if filter_set.issubset({str(t).lower() for t in (r.metadata or {}).get("tags", [])})
            ]

        results = results[:limit]

        # Format
        output = f"# Query Results ({len(results)})\n\n"
        output += f"**Query:** {query_text}\n"
        if session_id:
            output += f"**Session:** `{session_id[:20]}...`\n"
        if event_type:
            output += f"**Event Type:** {event_type}\n"
        output += "\n"

        if results:
            for i, node in enumerate(results[:limit], 1):
                ntype = (node.metadata or {}).get("event_type", "memory")
                preview = node.content[:200] + "..." if len(node.content) > 200 else node.content
                output += f"## {i}. [{ntype}] `{node.id[:12]}...`\n"
                output += f"{preview}\n"
                tags = (node.metadata or {}).get("tags", [])
                if tags:
                    output += f"*Tags: {', '.join(str(t) for t in tags[:5])}*\n"
                output += f"*Created: {node.created_at.isoformat()[:16]}*\n\n"
        else:
            output += "*No matching memories found.*\n"

        # Warn if embedding model is degraded (hash fallback active)
        try:
            from omega.graphs import is_embedding_degraded
            if is_embedding_degraded() and results:
                output += "\n**Note:** Semantic search is degraded (embedding model unavailable). Results are text-match only.\n"
        except Exception:
            pass

        logger.info(f"Query '{query_text[:30]}...' returned {len(results)} results")
        return output

    except Exception as e:
        logger.error(f"Query failed: {e}", exc_info=True)
        return f"# Query Error\n\n**Error:** {str(e)}\n"


def query_structured(
    query_text: str,
    limit: int = 10,
    session_id: Optional[str] = None,
    project: Optional[str] = None,
    event_type: Optional[str] = None,
    context_file: Optional[str] = None,
    context_tags: Optional[List[str]] = None,
    filter_tags: Optional[List[str]] = None,
    temporal_range: Optional[tuple] = None,
    entity_id: Optional[str] = None,
    agent_type: Optional[str] = None,
    surfacing_context: Optional["SurfacingContext"] = None,
) -> List[Dict[str, Any]]:
    """Query memories and return structured dicts (machine-readable)."""
    db = _get_store()

    try:
        effective_temporal = temporal_range or _infer_temporal_range(query_text)

        enhanced = query_text
        if event_type:
            enhanced = f"{event_type} {enhanced}"
        if project:
            enhanced = f"{Path(project).name} {enhanced}"

        results = db.query(
            enhanced,
            limit=limit * 2 if (filter_tags or entity_id or agent_type) else limit,
            session_id=session_id,
            context_file=context_file or "",
            context_tags=context_tags,
            temporal_range=effective_temporal,
            entity_id=entity_id,
            agent_type=agent_type,
            query_hint=event_type,
            surfacing_context=surfacing_context,
        )

        if event_type and results:
            results = [r for r in results if (r.metadata or {}).get("event_type") == event_type]

        # Hard filter by tags (AND logic — all specified tags must be present)
        if filter_tags and results:
            filter_set = {t.lower() for t in filter_tags}
            results = [
                r for r in results if filter_set.issubset({str(t).lower() for t in (r.metadata or {}).get("tags", [])})
            ]

        results = results[:limit]

        structured = []
        for node in results:
            structured.append(
                {
                    "id": node.id,
                    "content": node.content,
                    "event_type": (node.metadata or {}).get("event_type", "memory"),
                    "session_id": (node.metadata or {}).get("session_id", ""),
                    "created_at": node.created_at.isoformat() if node.created_at else "",
                    "tags": (node.metadata or {}).get("tags", []),
                    "metadata": node.metadata,
                    "relevance": getattr(node, "relevance", 0.0),
                }
            )

        return structured

    except Exception as e:
        logger.error(f"Structured query failed: {e}", exc_info=True)
        return []


# ---------------------------------------------------------------------------
# Public API -- Welcome / Session Bootstrap
# ---------------------------------------------------------------------------


def welcome(session_id: Optional[str] = None, project: Optional[str] = None) -> Dict[str, Any]:
    """Generate a session welcome briefing with relevant memories.

    Returns observation_prefix (grouped by type) and project_context
    for Claude to reference throughout the session.
    """
    db = _get_store()

    # Get recent high-value memories (decisions, lessons, preferences, errors)
    _HIGH_VALUE_TYPES = {"decision", "lesson_learned", "user_preference", "user_fact", "error_pattern"}
    recent = []
    try:
        candidates = db.get_recent(limit=100)
        for node in candidates:
            event_type = (node.metadata or {}).get("event_type", "")
            if event_type in _HIGH_VALUE_TYPES:
                recent.append(node)
                if len(recent) >= 15:
                    break
        # If no high-value memories found, fall back to most recent of any type
        if not recent:
            recent = candidates[:5]
    except Exception as e:
        logger.debug("Welcome recent memory filtering failed: %s", e)

    # Ensure user_preference and user_fact are always represented
    # These types have 95-98% never-accessed rates because recency-based
    # selection misses older entries. Direct type queries fix this.
    try:
        _entity_id = None
        if project:
            try:
                from omega.entity.engine import resolve_project_entity
                _entity_id = resolve_project_entity(project)
            except Exception:
                pass
        recent_ids = {n.id for n in recent}
        now_iso = datetime.now(timezone.utc).isoformat()
        for _type in ("user_preference", "user_fact"):
            type_nodes = db.get_by_type(_type, limit=5, entity_id=_entity_id)
            for node in type_nodes:
                if (node.metadata or {}).get("superseded"):
                    continue
                if node.id not in recent_ids:
                    recent.append(node)
                    recent_ids.add(node.id)
                # Bump access_count for all surfaced nodes (whether new or already in recent)
                try:
                    db._conn.execute(
                        "UPDATE memories SET access_count = access_count + 1, last_accessed = ? WHERE node_id = ?",
                        (now_iso, node.id),
                    )
                except Exception:
                    pass
        db._commit()
    except Exception as e:
        logger.debug("Welcome type-based enrichment failed: %s", e)

    # Sort by priority (desc) then recency
    recent.sort(
        key=lambda n: (
            (n.metadata or {}).get("priority", 3),
            n.created_at.isoformat() if n.created_at else "",
        ),
        reverse=True,
    )

    # Build observation_prefix — grouped by type
    observation_prefix = ""
    try:
        grouped: Dict[str, List[str]] = {}
        type_labels = {
            "user_preference": "User Preferences",
            "user_fact": "User Context",
            "decision": "Active Decisions",
            "lesson_learned": "Key Lessons",
            "error_pattern": "Known Pitfalls",
        }
        for n in recent[:15]:
            etype = (n.metadata or {}).get("event_type", "")
            label = type_labels.get(etype)
            if not label:
                continue
            # Prefer observation over raw content
            text = (n.metadata or {}).get("observation") or n.content[:150]
            if label not in grouped:
                grouped[label] = []
            if len(grouped[label]) < 5:
                grouped[label].append(text)

        if grouped:
            sections = []
            for label in ["User Preferences", "User Context", "Active Decisions", "Key Lessons", "Known Pitfalls"]:
                items = grouped.get(label, [])
                if items:
                    section = f"### {label}\n"
                    section += "\n".join(f"- {item}" for item in items)
                    sections.append(section)
            observation_prefix = "\n".join(sections)
    except Exception as e:
        logger.debug("Welcome observation_prefix failed: %s", e)

    # Build project_context
    project_context = ""
    if project:
        try:
            from omega.sqlite_store import SurfacingContext
            project_memories = db.query(
                Path(project).name,
                limit=5,
                project_path=project,
                scope="project",
                surfacing_context=SurfacingContext.SESSION_START,
            )
            if project_memories:
                items = []
                for m in project_memories[:5]:
                    text = (m.metadata or {}).get("observation") or m.content[:120]
                    items.append(f"- {text}")
                project_context = f"### Project: {Path(project).name}\n" + "\n".join(items)
        except Exception as e:
            logger.debug("Welcome project_context failed: %s", e)

    # Predictive prefetch (#5)
    if project:
        try:
            prefetched = db.prefetch_for_project(project)
            if prefetched:
                logger.debug("Prefetched %d memories for project %s", prefetched, project)
        except Exception as e:
            logger.debug("Welcome prefetch failed: %s", e)

    profile = get_profile()
    node_count = db.node_count()

    # Check for missing embedding model files (not backend activation — model is lazy-loaded)
    warnings = []
    from omega.graphs import _get_onnx_model_dir, has_onnx_runtime, has_sentence_transformers
    if has_onnx_runtime() and _get_onnx_model_dir() is None and not has_sentence_transformers():
        warnings.append(
            "Embedding model not found — semantic search will be disabled. "
            "Run 'omega setup' to download the model (~90 MB)."
        )

    result = {
        "greeting": f"Welcome back! You have {node_count} memories stored.",
        "recent_memories": [
            {
                "id": n.id,
                "content": n.content[:200],
                "type": (n.metadata or {}).get("event_type", "unknown"),
                "created_at": str(n.created_at),
                "relative_time": _relative_time(n.created_at),
            }
            for n in recent[:5]
        ],
        "observation_prefix": observation_prefix,
        "project_context": project_context,
        "profile": profile,
        "memory_count": node_count,
    }
    if warnings:
        result["warnings"] = warnings
    return result


def get_session_context(
    project: Optional[str] = None,
    exclude_session: Optional[str] = None,
    limit: int = 5,
) -> Dict[str, Any]:
    """Gather all data needed for session start briefing.

    Returns a dict with context_items (typed high-value memories),
    memory_count, health status, and last_capture_ago.
    """
    db = _get_store()
    node_count = db.node_count()

    # Health status
    health_status = "ok"
    try:
        health = db.check_memory_health()
        health_status = health.get("status", "ok")
    except Exception:
        health_status = "unknown"

    # Last capture time
    last_capture_ago = ""
    try:
        recent_all = db.get_recent(limit=1)
        if recent_all:
            last_capture_ago = _relative_time(recent_all[0].created_at)
    except Exception:
        pass

    # Gather typed high-value items for [CONTEXT] section
    _TYPE_TAG = {
        "user_preference": "PREF",
        "decision": "DECISION",
        "lesson_learned": "LESSON",
        "error_pattern": "PITFALL",
    }
    context_items: list[Dict[str, str]] = []
    try:
        candidates = db.get_recent(limit=100)
        seen_tags: Dict[str, int] = {}
        for node in candidates:
            etype = (node.metadata or {}).get("event_type", "")
            tag = _TYPE_TAG.get(etype)
            if not tag:
                continue
            if seen_tags.get(tag, 0) >= 2:
                continue
            text = (node.metadata or {}).get("observation") or node.content[:150]
            text = text.replace("\n", " ").strip()
            context_items.append({"tag": tag, "text": text})
            seen_tags[tag] = seen_tags.get(tag, 0) + 1
            if len(context_items) >= limit:
                break
    except Exception as e:
        logger.debug("get_session_context context_items failed: %s", e)

    return {
        "memory_count": node_count,
        "health_status": health_status,
        "last_capture_ago": last_capture_ago or "unknown",
        "context_items": context_items,
    }


# ---------------------------------------------------------------------------
# Public API -- Health & Status
# ---------------------------------------------------------------------------


def check_health(
    warn_mb: float = 350,
    critical_mb: float = 800,
    max_nodes: int = 10000,
) -> str:
    """Check OMEGA memory health. Returns formatted markdown."""
    db = _get_store()
    health = db.check_memory_health(warn_mb=warn_mb, critical_mb=critical_mb, max_nodes=max_nodes)

    status_label = health.get("status", "unknown").upper()
    output = "# OMEGA Health Check\n\n"
    output += f"**Status:** {status_label}\n"
    output += f"**Memory:** {health.get('memory_mb', 0):.1f} MB\n"
    output += f"**DB Size:** {health.get('db_size_mb', 0):.2f} MB\n"
    output += f"**Nodes:** {health.get('node_count', 0)}\n\n"

    warnings = health.get("warnings", [])
    if warnings:
        output += "## Warnings\n"
        for w in warnings:
            output += f"- {w}\n"
        output += "\n"

    recommendations = health.get("recommendations", [])
    if recommendations:
        output += "## Recommendations\n"
        for r in recommendations:
            output += f"- {r}\n"
        output += "\n"

    return output


def status() -> Dict[str, Any]:
    """Return a machine-readable health/status dict."""
    db = _get_store()
    try:
        health = db.check_memory_health()
        return {
            "ok": health.get("status") == "healthy",
            "status": health.get("status", "unknown"),
            "node_count": health.get("node_count", 0),
            "memory_mb": health.get("memory_mb", 0),
            "db_size_mb": health.get("db_size_mb", 0),
            "warnings": health.get("warnings", []),
            "store_path": str(OMEGA_HOME),
            "backend": "sqlite",
            "vec_enabled": health.get("usage", {}).get("vec_enabled", False),
        }
    except Exception as e:
        logger.error(f"Status check failed: {e}", exc_info=True)
        return {"ok": False, "error": str(e)}


def get_dedup_stats() -> Dict[str, Any]:
    """Return deduplication statistics."""
    db = _get_store()
    return {
        "content_dedup_skips": db.stats.get("content_dedup_skips", 0),
        "memory_evolutions": db.stats.get("memory_evolutions", 0),
        "embedding_dedup_skips": db.stats.get("embedding_dedup_skips", 0),
        "node_count": db.node_count(),
    }


# ---------------------------------------------------------------------------
# Public API -- Export / Import
# ---------------------------------------------------------------------------


def export_memories(filepath: str) -> str:
    """Export all OMEGA memories to a file."""
    db = _get_store()
    result = db.export_to_file(Path(filepath))

    output = "# OMEGA Export Complete\n\n"
    output += f"**File:** {result.get('filepath', filepath)}\n"
    output += f"**Nodes:** {result.get('node_count', 0)}\n"
    output += f"**Sessions:** {result.get('session_count', 0)}\n"
    output += f"**Size:** {result.get('file_size_kb', 0):.1f} KB\n"
    output += f"**Exported:** {result.get('exported_at', 'now')}\n"

    logger.info(f"Exported OMEGA memories to {filepath}")
    return output


def import_memories(filepath: str, clear_existing: bool = True) -> str:
    """Import OMEGA memories from a file."""
    db = _get_store()
    result = db.import_from_file(Path(filepath), clear_existing=clear_existing)

    output = "# OMEGA Import Complete\n\n"
    output += f"**File:** {result.get('filepath', filepath)}\n"
    output += f"**Nodes Imported:** {result.get('node_count', 0)}\n"
    output += f"**Sessions:** {result.get('session_count', 0)}\n"
    output += f"**Cleared Existing:** {'Yes' if clear_existing else 'No'}\n"

    logger.info(f"Imported OMEGA memories from {filepath}")
    return output


# ---------------------------------------------------------------------------
# Public API -- Deduplication
# ---------------------------------------------------------------------------


def deduplicate(
    event_type: Optional[str] = "lesson_learned",
    similarity_threshold: float = 0.80,
    dry_run: bool = False,
    session_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Find and merge duplicate memories using Jaccard similarity."""
    db = _get_store()
    result: Dict[str, Any] = {
        "event_type": event_type or "all",
        "similarity_threshold": similarity_threshold,
        "dry_run": dry_run,
        "groups_found": 0,
        "duplicates_removed": 0,
        "memories_kept": 0,
        "details": [],
    }

    # Gather candidates
    if event_type:
        candidates = db.get_by_type(event_type, limit=500)
    else:
        candidates = db.get_recent(limit=500)

    if session_id:
        candidates = [n for n in candidates if (n.metadata or {}).get("session_id") == session_id]

    if len(candidates) < 2:
        result["message"] = f"Only {len(candidates)} memories found, nothing to deduplicate."
        return result

    # Build word sets
    def _norm(text: str) -> set:
        return {re.sub(r"[^\w]", "", w) for w in text.lower().split() if len(w) > 3}

    node_words = [(node, _norm(node.content)) for node in candidates]

    # Union-find style grouping
    merged_into: Dict[str, str] = {}
    groups: Dict[str, list] = {}

    for i in range(len(node_words)):
        node_i, words_i = node_words[i]
        if node_i.id in merged_into or not words_i:
            continue

        group = [node_i]
        for j in range(i + 1, len(node_words)):
            node_j, words_j = node_words[j]
            if node_j.id in merged_into or not words_j:
                continue
            intersection = len(words_i & words_j)
            union = len(words_i | words_j)
            if union and (intersection / union) >= similarity_threshold:
                group.append(node_j)
                merged_into[node_j.id] = node_i.id

        if len(group) > 1:
            groups[node_i.id] = group

    result["groups_found"] = len(groups)

    for _rep_id, group in groups.items():
        group.sort(key=lambda n: len(n.content), reverse=True)
        keeper = group[0]
        duplicates = group[1:]
        total_access = sum(getattr(n, "access_count", 0) or 0 for n in group)

        detail = {
            "kept": {
                "id": keeper.id[:12],
                "content_preview": keeper.content[:100],
                "access_count": total_access,
            },
            "removed": [{"id": n.id[:12], "content_preview": n.content[:80]} for n in duplicates],
            "group_size": len(group),
        }
        result["details"].append(detail)

        if not dry_run:
            db.update_node(keeper.id, access_count=total_access)
            for dup in duplicates:
                try:
                    db.delete_node(dup.id)
                    result["duplicates_removed"] += 1
                except Exception as e:
                    logger.warning(f"Failed to remove duplicate {dup.id[:12]}: {e}")
            result["memories_kept"] += 1

    if not dry_run and result["duplicates_removed"] > 0:
        logger.info(
            f"Deduplication complete: {result['groups_found']} groups, "
            f"{result['duplicates_removed']} removed, "
            f"{result['memories_kept']} kept"
        )

    return result


# ---------------------------------------------------------------------------
# Public API -- Preferences
# ---------------------------------------------------------------------------


def extract_preferences(text: str) -> Dict[str, Any]:
    """Extract user preferences from free text and store them."""
    try:
        from omega.preferences import PreferenceExtractor

        extractor = PreferenceExtractor()
        prefs = extractor.extract(text)
        stored = []
        for pref in prefs:
            auto_capture(
                content=f"[Preference] {pref.get('key', 'unknown')}: {pref.get('value', text[:100])}",
                event_type=AutoCaptureEventType.USER_PREFERENCE,
                metadata={"preference_key": pref.get("key"), "preference_value": pref.get("value")},
            )
            stored.append({"key": pref.get("key"), "stored": True})
        return {"success": True, "preferences": stored, "count": len(stored)}
    except ImportError:
        auto_capture(
            content=f"[Preference] {text[:500]}",
            event_type=AutoCaptureEventType.USER_PREFERENCE,
            metadata={"source": "raw_text"},
        )
        return {"success": True, "preferences": [{"key": "raw", "stored": True}], "count": 1}
    except Exception as e:
        logger.error(f"Preference extraction failed: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


def list_preferences() -> List[Dict[str, Any]]:
    """List stored user preferences."""
    db = _get_store()
    try:
        nodes = db.get_by_type(AutoCaptureEventType.USER_PREFERENCE, limit=100)
        return [
            {
                "id": n.id,
                "content": n.content,
                "created_at": n.created_at.isoformat() if n.created_at else "",
                "metadata": n.metadata or {},
            }
            for n in nodes
        ]
    except Exception as e:
        logger.error(f"list_preferences failed: {e}", exc_info=True)
        return []


# ---------------------------------------------------------------------------
# Public API -- Profile
# ---------------------------------------------------------------------------


def get_profile() -> Dict[str, Any]:
    """Get the user profile from the OMEGA home directory, augmented with preference memories."""
    profile_path = OMEGA_HOME / "profile.json"
    profile: Dict[str, Any] = {}
    try:
        if profile_path.exists():
            with open(profile_path, "r") as f:
                profile = json.loads(f.read())
    except Exception as e:
        logger.debug(f"Failed to load profile: {e}")
    # Augment with preference memories
    try:
        store = _get_store()
        prefs = store.get_by_type("user_preference", limit=20)
        if prefs:
            profile["preferences_from_memory"] = [
                {
                    "content": m.content,
                    "created": m.created_at.isoformat() if hasattr(m.created_at, "isoformat") else str(m.created_at),
                }
                for m in prefs
            ]
    except Exception as e:
        logger.debug(f"Failed to load preference memories: {e}")
    return profile


def save_profile(profile: Dict[str, Any]) -> bool:
    """Persist the user profile to disk (atomic write via temp+rename)."""
    profile_path = OMEGA_HOME / "profile.json"
    try:
        profile_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        import tempfile

        fd, tmp_path = tempfile.mkstemp(dir=profile_path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                f.write(json.dumps(profile, indent=2))
            os.replace(tmp_path, profile_path)
        except BaseException:
            os.unlink(tmp_path)
            raise
        return True
    except Exception as e:
        logger.error(f"Failed to save profile: {e}", exc_info=True)
        return False


# ---------------------------------------------------------------------------
# Public API -- Cross-session lessons
# ---------------------------------------------------------------------------


def get_cross_session_lessons(
    task: Optional[str] = None,
    project_path: Optional[str] = None,
    exclude_session: Optional[str] = None,
    limit: int = 5,
    agent_type: Optional[str] = None,
    context_file: Optional[str] = None,
    context_tags: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Retrieve top lessons from ALL past sessions for injection."""
    db = _get_store()
    lessons: List[Dict[str, Any]] = []
    seen_content: set = set()
    lesson_sessions: Dict[str, set] = {}

    try:
        if task and (context_file or context_tags):
            # Use full query() for contextual re-ranking when context is available
            enhanced = f"lesson_learned {task}"
            if project_path:
                enhanced = f"{Path(project_path).name} {enhanced}"
            raw = db.query(
                enhanced,
                limit=limit * 3,
                context_file=context_file or "",
                context_tags=context_tags,
                project_path=project_path or "",
            )
            nodes = [r for r in raw if (r.metadata or {}).get("event_type") == "lesson_learned"]
        elif task:
            nodes = db.query_by_type(query=task, event_type="lesson_learned", limit=limit * 3)
        else:
            nodes = db.get_by_type("lesson_learned", limit=limit * 3)

        for node in nodes:
            meta = node.metadata or {}
            if exclude_session and meta.get("session_id") == exclude_session:
                continue
            if agent_type and meta.get("agent_type") != agent_type:
                continue

            key = node.content[:80].lower()
            node_session = meta.get("session_id", "")

            if key in seen_content:
                if node_session and key in lesson_sessions:
                    lesson_sessions[key].add(node_session)
                continue

            seen_content.add(key)
            lesson_sessions[key] = {node_session} if node_session else set()

            lessons.append(
                {
                    "content": node.content,
                    "source": "omega",
                    "lesson_id": meta.get("lesson_id") or node.id,
                    "session_id": node_session,
                    "access_count": getattr(node, "access_count", 0) or 0,
                    "created_at": node.created_at.isoformat() if node.created_at else "",
                    "verified_count": 0,
                    "_key": key,
                }
            )
    except Exception as e:
        logger.debug(f"Lesson query failed: {e}")

    for lesson in lessons:
        key = lesson.get("_key", "")
        session_count = len(lesson_sessions.get(key, set()))
        if session_count > 1:
            lesson["verified_count"] = max(lesson.get("verified_count", 0), session_count)
        lesson["verified"] = lesson.get("verified_count", 0) > 0
        lesson.pop("_key", None)

    lessons.sort(
        key=lambda lesson: (lesson.get("verified_count", 0), lesson.get("access_count", 0)),
        reverse=True,
    )

    return lessons[:limit]


# ---------------------------------------------------------------------------
# Public API -- Constraint Enforcement
# ---------------------------------------------------------------------------

CONSTRAINTS_DIR = OMEGA_HOME / "constraints"


def _load_constraints(project: Optional[str] = None) -> List[Dict[str, Any]]:
    """Load constraint rules for a project from ~/.omega/constraints/.

    Loads global.json first, then <project-name>.json if project is given.
    Returns merged list of rule dicts.
    """
    rules: List[Dict[str, Any]] = []
    if not CONSTRAINTS_DIR.exists():
        return rules

    # Global constraints
    global_file = CONSTRAINTS_DIR / "global.json"
    if global_file.exists():
        try:
            data = json.loads(global_file.read_text())
            for r in data.get("rules", []):
                r["source"] = "global"
                rules.append(r)
        except Exception as e:
            logger.debug(f"Failed to load global constraints: {e}")

    # Project-specific constraints
    if project:
        proj_name = Path(project).name
        proj_file = CONSTRAINTS_DIR / f"{proj_name}.json"
        if proj_file.exists():
            try:
                data = json.loads(proj_file.read_text())
                for r in data.get("rules", []):
                    r["source"] = proj_name
                    rules.append(r)
            except Exception as e:
                logger.debug(f"Failed to load {proj_name} constraints: {e}")

    return rules


def check_constraints(file_path: str, project: Optional[str] = None) -> List[Dict[str, Any]]:
    """Check a file path against loaded constraint rules.

    Returns list of matching constraints with severity and message.
    """
    import fnmatch

    rules = _load_constraints(project)
    if not rules:
        return []

    matches = []
    filename = os.path.basename(file_path)

    for rule in rules:
        pattern = rule.get("pattern", "")
        if not pattern:
            continue
        # Match against filename or full path
        if fnmatch.fnmatch(filename, pattern) or fnmatch.fnmatch(file_path, pattern):
            matches.append(
                {
                    "pattern": pattern,
                    "constraint": rule.get("constraint", ""),
                    "severity": rule.get("severity", "warn"),
                    "source": rule.get("source", "unknown"),
                }
            )

    return matches


def list_constraints(project: Optional[str] = None) -> Dict[str, Any]:
    """List all loaded constraint rules for a project."""
    rules = _load_constraints(project)
    return {
        "count": len(rules),
        "rules": rules,
        "constraints_dir": str(CONSTRAINTS_DIR),
    }


def save_constraints(
    rules: List[Dict[str, Any]],
    project: Optional[str] = None,
) -> Dict[str, Any]:
    """Save constraint rules to the appropriate file.

    If project is given, saves to <project-name>.json, else global.json.
    """
    CONSTRAINTS_DIR.mkdir(parents=True, exist_ok=True, mode=0o700)

    if project:
        target = CONSTRAINTS_DIR / f"{Path(project).name}.json"
    else:
        target = CONSTRAINTS_DIR / "global.json"

    # Clean source field from rules before saving
    clean_rules = []
    for r in rules:
        clean = {k: v for k, v in r.items() if k != "source"}
        clean_rules.append(clean)

    data = {"rules": clean_rules}
    target.write_text(json.dumps(data, indent=2))

    return {"saved": str(target), "count": len(clean_rules)}


# ---------------------------------------------------------------------------
# Public API -- Cross-project Learning
# ---------------------------------------------------------------------------


def get_cross_project_lessons(
    task: Optional[str] = None,
    exclude_project: Optional[str] = None,
    exclude_session: Optional[str] = None,
    limit: int = 5,
    agent_type: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Retrieve lessons from ALL projects (cross-project knowledge transfer).

    Unlike get_cross_session_lessons which may filter by project,
    this explicitly queries without project scope to find patterns
    that recur across different codebases.
    """
    db = _get_store()
    lessons: List[Dict[str, Any]] = []
    seen_content: set = set()
    project_sets: Dict[str, set] = {}

    try:
        if task:
            nodes = db.query_by_type(query=task, event_type="lesson_learned", limit=limit * 4)
        else:
            nodes = db.get_by_type("lesson_learned", limit=limit * 4)

        for node in nodes:
            meta = node.metadata or {}
            node_project = meta.get("project", "")

            if exclude_session and meta.get("session_id") == exclude_session:
                continue
            if exclude_project and node_project == exclude_project:
                continue
            if agent_type and meta.get("agent_type") != agent_type:
                continue

            key = node.content[:80].lower()

            if key in seen_content:
                if node_project and key in project_sets:
                    project_sets[key].add(node_project)
                continue

            seen_content.add(key)
            project_sets[key] = {node_project} if node_project else set()

            lessons.append(
                {
                    "content": node.content,
                    "source_project": node_project,
                    "lesson_id": meta.get("lesson_id") or node.id,
                    "session_id": meta.get("session_id", ""),
                    "access_count": getattr(node, "access_count", 0) or 0,
                    "created_at": node.created_at.isoformat() if node.created_at else "",
                    "projects_seen": 1,
                    "_key": key,
                }
            )
    except Exception as e:
        logger.debug(f"Cross-project lesson query failed: {e}")

    # Enrich with cross-project counts
    for lesson in lessons:
        key = lesson.get("_key", "")
        proj_count = len(project_sets.get(key, set()))
        lesson["projects_seen"] = max(1, proj_count)
        lesson["cross_project"] = proj_count > 1
        lesson.pop("_key", None)

    # Sort by cross-project occurrence, then access count
    lessons.sort(
        key=lambda lesson: (lesson.get("projects_seen", 0), lesson.get("access_count", 0)),
        reverse=True,
    )

    return lessons[:limit]


# ---------------------------------------------------------------------------
# Public API -- Reingest (legacy JSONL → SQLite)
# ---------------------------------------------------------------------------


def reingest(
    store_path: Optional[Path] = None,
    batch_size: int = 50,
    skip_types: Optional[set] = None,
) -> Dict[str, Any]:
    """Bulk-load JSONL store entries into SQLite.

    Reads every line from store.jsonl and inserts into the SQLite database.
    Content-hash dedup prevents duplicates automatically.
    """
    db = _get_store()
    src = store_path or (OMEGA_HOME / "store.jsonl")

    if not src.exists():
        return {"error": f"Store file not found: {src}", "ingested": 0}

    skip_types = skip_types or set()
    stats = {"ingested": 0, "skipped": 0, "duplicates": 0, "errors": 0, "total": 0}

    logger.info(f"Reingesting from {src}")

    from omega.crypto import decrypt_line

    with open(src, "r") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            stats["total"] += 1

            try:
                entry = json.loads(decrypt_line(line))
            except Exception:
                stats["errors"] += 1
                continue

            content = entry.get("content", "").strip()
            if not content:
                stats["skipped"] += 1
                continue

            meta = entry.get("metadata", {})
            event_type = meta.get("event_type", "memory")

            if event_type in skip_types:
                stats["skipped"] += 1
                continue

            session_id = meta.get("session_id")
            ttl = TTLCategory.for_event_type(event_type)

            try:
                db.store(
                    content=content[:2000],
                    session_id=session_id,
                    metadata=meta,
                    ttl_seconds=ttl,
                    skip_inference=True,
                )
                stats["ingested"] += 1
            except Exception as e:
                stats["errors"] += 1
                if stats["errors"] <= 5:
                    logger.warning(f"Reingest error line {line_num}: {e}")

            if stats["ingested"] > 0 and stats["ingested"] % batch_size == 0:
                logger.info(f"  Progress: {stats['ingested']} ingested, {stats['total']} processed")

    logger.info(
        f"Reingest complete: {stats['ingested']} ingested, "
        f"{stats['duplicates']} duplicates, {stats['errors']} errors "
        f"out of {stats['total']} entries"
    )
    return stats


# ---------------------------------------------------------------------------
# Public API -- Feedback
# ---------------------------------------------------------------------------


def record_feedback(
    memory_id: str,
    rating: str,
    reason: Optional[str] = None,
) -> Dict[str, Any]:
    """Record feedback on a surfaced memory."""
    db = _get_store()
    return db.record_feedback(node_id=memory_id, rating=rating, reason=reason)


# ---------------------------------------------------------------------------
# Public API -- Session management
# ---------------------------------------------------------------------------


def clear_session(session_id: str) -> Dict[str, Any]:
    """Clear all memories for a session."""
    db = _get_store()
    count = db.clear_session(session_id)
    logger.info(f"Cleared session {session_id[:16]}: {count} memories removed")
    return {"session_id": session_id, "removed": count}


# ---------------------------------------------------------------------------
# Public API -- Batch operations
# ---------------------------------------------------------------------------


def batch_store(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Store multiple memories efficiently."""
    db = _get_store()
    ids = db.batch_store(items)
    return {"ids": ids, "count": len(ids)}


# ---------------------------------------------------------------------------
# Public API -- Similar memories
# ---------------------------------------------------------------------------


def find_similar_memories(memory_id: str, limit: int = 5) -> str:
    """Find memories similar to a given memory via vector search."""
    db = _get_store()
    node = db.get_node(memory_id)
    if node is None:
        return f"Memory `{memory_id}` not found."
    embedding = db.get_embedding(memory_id)
    if embedding is None:
        return f"No embedding found for `{memory_id[:12]}`. Vector search unavailable."
    # limit+1 because the source memory will be in results
    results = db.find_similar(embedding, limit=limit + 1)
    # Filter out the source memory itself
    results = [r for r in results if r.id != memory_id][:limit]
    # Format output
    output = f"# Similar Memories ({len(results)})\n\n"
    output += f"**Source:** `{memory_id[:12]}` — {node.content[:100]}\n\n"
    for i, r in enumerate(results, 1):
        ntype = (r.metadata or {}).get("event_type", "memory")
        preview = r.content[:200]
        output += f"## {i}. [{ntype}] `{r.id[:12]}` (similarity: {r.relevance:.2f})\n"
        output += f"{preview}\n\n"
    if not results:
        output += "*No similar memories found.*\n"
    return output


# ---------------------------------------------------------------------------
# Public API -- Timeline
# ---------------------------------------------------------------------------


def timeline(days: int = 7, limit_per_day: int = 10) -> str:
    """Show memory timeline grouped by day."""
    db = _get_store()
    data = db.get_timeline(days=days, limit_per_day=limit_per_day)
    if not data:
        return f"# Memory Timeline\n\nNo memories in the last {days} days."
    total = sum(len(v) for v in data.values())
    output = f"# Memory Timeline ({total} memories, last {days} days)\n\n"
    for day in sorted(data.keys(), reverse=True):
        memories = data[day]
        output += f"## {day} ({len(memories)} memories)\n\n"
        for m in memories:
            etype = (m.metadata or {}).get("event_type", "memory")
            tags = (m.metadata or {}).get("tags", [])
            tag_str = f" [{', '.join(str(t) for t in tags[:3])}]" if tags else ""
            preview = m.content[:120].replace("\n", " ")
            output += f"- **[{etype}]** {preview}{tag_str}\n"
            output += f"  `{m.id[:12]}` · {m.created_at.strftime('%H:%M')}\n"
        output += "\n"
    return output


# ---------------------------------------------------------------------------
# Public API -- Consolidation
# ---------------------------------------------------------------------------


def _auto_backup_before_consolidate():
    """Create a backup before consolidation (rotate to keep last 3)."""
    db_path = OMEGA_HOME / "omega.db"
    if not db_path.exists():
        return
    try:
        import sqlite3
        from omega.crypto import secure_connect

        backups_dir = OMEGA_HOME / "backups"
        backups_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        backup_path = backups_dir / f"pre-consolidate-{timestamp}.db"
        src = sqlite3.connect(str(db_path))
        dst = secure_connect(backup_path)
        src.backup(dst)
        dst.close()
        src.close()
        logger.info(f"Pre-consolidation backup: {backup_path}")
        # Rotate — keep only last 3
        backups = sorted(backups_dir.glob("pre-consolidate-*.db"), key=lambda p: p.stat().st_mtime, reverse=True)
        for old in backups[3:]:
            old.unlink()
    except Exception as e:
        logger.warning(f"Auto-backup before consolidation failed: {e}")


def consolidate(prune_days: int = 30, max_summaries: int = 50) -> str:
    """Run memory consolidation: prune stale entries, cap summaries, clean edges.

    Returns formatted markdown report.
    """
    _auto_backup_before_consolidate()
    db = _get_store()
    before = db.node_count()
    stats = db.consolidate(prune_days=prune_days, max_summaries=max_summaries)
    after = stats.get("node_count_after", before)
    removed = before - after

    output = "# Memory Consolidation Report\n\n"
    output += f"**Before:** {before} memories\n"
    output += f"**After:** {after} memories\n"
    output += f"**Removed:** {removed} total\n\n"
    output += "## Breakdown\n\n"
    output += f"- **Stale (0 access, >{prune_days}d old):** {stats.get('pruned_stale', 0)}\n"
    output += f"- **Session summaries (beyond cap of {max_summaries}):** {stats.get('pruned_summaries', 0)}\n"
    output += f"- **Orphaned edges:** {stats.get('pruned_edges', 0)}\n"

    if removed == 0:
        output += "\n*Nothing to consolidate — memory store is clean.*\n"
    else:
        logger.info(f"Consolidation: removed {removed} memories ({stats})")

    return output


# ---------------------------------------------------------------------------
# Public API -- Memory Compaction
# ---------------------------------------------------------------------------


def _smart_extract(cluster) -> str:
    """Extract diverse, information-dense sentences from a cluster of memories.

    Scores sentences by: unique-word count (IDF-like), sentence length
    (diminishing returns), presence of proper nouns / code tokens, and
    cross-memory term frequency (words appearing in 2+ cluster members
    are more generalizable — ALMA-inspired strategy extraction).

    For large clusters (5+), extracts a strategy header from the most
    common bigram theme across cluster members.

    Skips near-duplicate sentences (Jaccard > 0.7).
    Orders selected sentences chronologically by source memory.
    Returns consolidated text capped at 1000 chars.
    """
    # Build cross-memory word frequency map (words appearing in 2+ members)
    from collections import Counter
    word_to_members: dict = {}  # word -> set of node indices
    for idx, node in enumerate(cluster):
        for w in set(node.content.lower().split()):
            if len(w) > 3:
                word_to_members.setdefault(w, set()).add(idx)
    cross_freq_words = {w for w, members in word_to_members.items() if len(members) >= 2}

    # Collect all sentences with source metadata
    all_sentences = []  # [(sentence, density_score, created_at)]
    seen_keys: set = set()

    for node in cluster:
        created = node.created_at.isoformat() if node.created_at else ""
        for sentence in re.split(r"(?<=[.!?])\s+", node.content):
            sentence = sentence.strip()
            if len(sentence) < 15:
                continue
            key = " ".join(sentence.lower().split())[:100]
            if key in seen_keys:
                continue
            seen_keys.add(key)

            words = sentence.split()
            unique_words = len(set(w.lower() for w in words if len(w) > 3))

            # Proper nouns / capitalized words (not sentence-start)
            proper_nouns = len([w for w in words[1:] if w[0].isupper()]) if len(words) > 1 else 0

            # Code tokens: backtick spans, paths, CamelCase
            code_tokens = len(re.findall(r"`[^`]+`|/[\w/.]+|\b[A-Z][a-z]+[A-Z]\w*\b", sentence))

            # Diminishing returns on length
            length_score = min(len(sentence), 200) / 200.0

            # Cross-memory term frequency boost (ALMA-inspired)
            cross_freq = sum(1 for w in words if w.lower() in cross_freq_words)

            density = (unique_words * 1.0 + proper_nouns * 1.5 + code_tokens * 2.0
                       + length_score * 3.0 + cross_freq * 0.8)
            all_sentences.append((sentence, density, created))

    if not all_sentences:
        return ""

    # Sort by density (highest first)
    all_sentences.sort(key=lambda x: x[1], reverse=True)

    # Select top-K diverse sentences
    selected = []
    for sentence, _score, created in all_sentences:
        # Check diversity against already selected
        is_diverse = all(_jaccard(sentence.lower(), sel[0].lower(), min_word_len=3) < 0.7 for sel in selected)
        if is_diverse:
            selected.append((sentence, created))
            if len(selected) >= 8:  # Max sentences to consider
                break

    # Order chronologically by source memory created_at
    selected.sort(key=lambda x: x[1])

    # Build consolidated text (cap at 1000 chars)
    consolidated = " ".join(s for s, _ in selected)

    # Strategy header for large clusters (5+ members): extract common bigram theme
    if len(cluster) >= 5:
        bigram_counter: Counter = Counter()
        for node in cluster:
            words = [w.lower() for w in node.content.split() if len(w) > 3]
            for i in range(len(words) - 1):
                bigram_counter[(words[i], words[i + 1])] += 1
        if bigram_counter:
            top_bigram, top_count = bigram_counter.most_common(1)[0]
            if top_count >= 3:  # Only if bigram appears in 3+ members
                theme = f"{top_bigram[0]} {top_bigram[1]}"
                consolidated = f"Strategy: {theme}. {consolidated}"

    if len(consolidated) > 1000:
        consolidated = consolidated[:997] + "..."

    return consolidated


def compact(
    event_type: str = "lesson_learned",
    similarity_threshold: float = 0.60,
    min_cluster_size: int = 3,
    dry_run: bool = False,
) -> str:
    """Compact clusters of related memories into consolidated knowledge nodes.

    Unlike deduplicate() which removes exact/near duplicates, compact() finds
    clusters of semantically related memories and creates new summary nodes
    that consolidate the knowledge, marking originals as superseded.

    Returns formatted markdown report.
    """
    db = _get_store()
    candidates = db.get_by_type(event_type, limit=500)

    if len(candidates) < min_cluster_size:
        return (
            f"# Memory Compaction\n\n"
            f"Only {len(candidates)} `{event_type}` memories found "
            f"(minimum cluster size: {min_cluster_size}). Nothing to compact.\n"
        )

    # Build word sets for Jaccard clustering
    def _norm(text: str) -> set:
        return {re.sub(r"[^\w]", "", w) for w in text.lower().split() if len(w) > 3}

    node_words = [(node, _norm(node.content)) for node in candidates]

    # Union-find style clustering
    assigned: set = set()
    clusters: List[List] = []

    for i in range(len(node_words)):
        node_i, words_i = node_words[i]
        if node_i.id in assigned or not words_i:
            continue

        cluster = [node_i]
        assigned.add(node_i.id)

        for j in range(i + 1, len(node_words)):
            node_j, words_j = node_words[j]
            if node_j.id in assigned or not words_j:
                continue
            intersection = len(words_i & words_j)
            union = len(words_i | words_j)
            if union and (intersection / union) >= similarity_threshold:
                cluster.append(node_j)
                assigned.add(node_j.id)

        if len(cluster) >= min_cluster_size:
            clusters.append(cluster)

    if not clusters:
        return (
            f"# Memory Compaction\n\n"
            f"No clusters found with >= {min_cluster_size} similar `{event_type}` memories "
            f"at {similarity_threshold:.0%} similarity. Store is already compact.\n"
        )

    # Build report and optionally perform compaction
    output = f"# Memory Compaction {'(DRY RUN)' if dry_run else 'Report'}\n\n"
    output += f"**Event type:** {event_type}\n"
    output += f"**Similarity threshold:** {similarity_threshold:.0%}\n"
    output += f"**Clusters found:** {len(clusters)}\n\n"

    total_compacted = 0
    total_created = 0

    for ci, cluster in enumerate(clusters, 1):
        # Sort by content length (longest first — most information)
        cluster.sort(key=lambda n: len(n.content), reverse=True)

        consolidated = _smart_extract(cluster)

        # Merge tags from all cluster members
        merged_tags: set = set()
        total_access = 0
        for node in cluster:
            merged_tags.update(str(t) for t in (node.metadata or {}).get("tags", []))
            total_access += getattr(node, "access_count", 0) or 0

        output += f"## Cluster {ci} ({len(cluster)} memories)\n\n"
        output += f"**Summary:** {consolidated[:200]}...\n"
        for node in cluster[:5]:
            preview = node.content[:80]
            output += f"- `{node.id[:12]}`: {preview}\n"
        if len(cluster) > 5:
            output += f"- ... and {len(cluster) - 5} more\n"
        output += "\n"

        if not dry_run:
            # Prefix consolidated content to distinguish from originals (avoids dedup)
            compact_header = f"[Consolidated from {len(cluster)} memories] "
            compact_content = compact_header + consolidated

            # Create the consolidated node with quality metadata
            # Quality scale: 1.0 (min cluster) to 3.0 (10+ members)
            consolidation_quality = min(3.0, 1.0 + (len(cluster) - min_cluster_size) * 0.3)
            meta = {
                "event_type": event_type,
                "source": "compaction",
                "compacted_from": [n.id for n in cluster],
                "compacted_count": len(cluster),
                "tags": sorted(merged_tags)[:15],
                "consolidation_quality": round(consolidation_quality, 2),
            }
            new_id = db.store(
                content=compact_content,
                metadata=meta,
                ttl_seconds=TTLCategory.for_event_type(event_type),
                skip_inference=True,  # Bypass embedding dedup
            )
            db.update_node(new_id, access_count=total_access)

            # Mark originals as superseded
            for node in cluster:
                nmeta = dict(node.metadata or {})
                nmeta["superseded"] = True
                nmeta["superseded_by"] = new_id
                nmeta["compacted_at"] = datetime.now(timezone.utc).isoformat()
                db.update_node(node.id, metadata=nmeta)

            total_compacted += len(cluster)
            total_created += 1
            output += f"**Created:** `{new_id[:12]}` | **Superseded:** {len(cluster)} memories\n\n"

    output += "---\n"
    if dry_run:
        output += f"**Would compact:** {sum(len(c) for c in clusters)} memories into {len(clusters)} nodes\n"
    else:
        output += f"**Compacted:** {total_compacted} memories into {total_created} consolidated nodes\n"

    return output


# ---------------------------------------------------------------------------
# Public API -- Graph Traversal
# ---------------------------------------------------------------------------


def traverse(
    memory_id: str,
    max_hops: int = 2,
    min_weight: float = 0.0,
    edge_types: Optional[List[str]] = None,
) -> str:
    """Traverse the relationship graph from a starting memory.

    Walks the `related` edges table up to max_hops, returning all
    connected memories with their hop distance and edge weight.

    Returns formatted markdown string.
    """
    db = _get_store()
    node = db.get_node(memory_id)
    if node is None:
        return f"Memory `{memory_id}` not found."

    results = db.get_related_chain(
        start_id=memory_id,
        max_hops=max_hops,
        min_weight=min_weight,
        edge_types=edge_types,
    )

    output = f"# Graph Traversal ({len(results)} connected memories)\n\n"
    output += f"**Start:** `{memory_id[:12]}` — {node.content[:100]}\n"
    output += f"**Max hops:** {max_hops}\n\n"

    if not results:
        output += "*No connected memories found.*\n"
        return output

    current_hop = 0
    for r in results:
        if r["hop"] != current_hop:
            current_hop = r["hop"]
            output += f"## Hop {current_hop}\n\n"

        etype = (r.get("metadata") or {}).get("event_type", "memory")
        preview = r["content"][:200]
        output += f"- **[{etype}]** `{r['node_id'][:12]}` (weight: {r['weight']:.2f}, edge: {r['edge_type']})\n"
        output += f"  {preview}\n\n"

    return output


# ---------------------------------------------------------------------------
# Public API -- Phrase Search
# ---------------------------------------------------------------------------


def phrase_search(
    phrase: str,
    limit: int = 10,
    event_type: Optional[str] = None,
    project: Optional[str] = None,
    case_sensitive: bool = False,
) -> str:
    """Search memories for exact phrase matches using FTS5.

    Returns formatted markdown string.
    """
    db = _get_store()
    try:
        results = db.phrase_search(
            phrase=phrase,
            limit=limit,
            event_type=event_type,
            case_sensitive=case_sensitive,
            project_path=project or "",
        )

        output = f"# Phrase Search Results ({len(results)})\n\n"
        output += f'**Phrase:** "{phrase}"\n'
        if event_type:
            output += f"**Event Type:** {event_type}\n"
        output += "\n"

        if results:
            for i, node in enumerate(results[:limit], 1):
                ntype = (node.metadata or {}).get("event_type", "memory")
                preview = node.content[:200] + "..." if len(node.content) > 200 else node.content
                output += f"## {i}. [{ntype}] `{node.id[:12]}...`\n"
                output += f"{preview}\n"
                tags = (node.metadata or {}).get("tags", [])
                if tags:
                    output += f"*Tags: {', '.join(str(t) for t in tags[:5])}*\n"
                output += f"*Created: {node.created_at.isoformat()[:16]}*\n\n"
        else:
            output += "*No matching memories found.*\n"

        return output

    except Exception as e:
        logger.error(f"Phrase search failed: {e}", exc_info=True)
        return f"# Phrase Search Error\n\n**Error:** {str(e)}\n"


# ---------------------------------------------------------------------------
# Public API -- Stats
# ---------------------------------------------------------------------------


def type_stats() -> Dict[str, int]:
    """Get memory counts grouped by event type."""
    db = _get_store()
    return db.get_type_stats()


def session_stats() -> Dict[str, int]:
    """Get memory counts grouped by session ID."""
    db = _get_store()
    return db.get_session_stats()


def access_rate_stats() -> Dict[str, Any]:
    """Get access rate breakdown: never-accessed count, by-type, top accessed."""
    db = _get_store()
    total = db.node_count()

    zero_access = db._conn.execute(
        "SELECT COUNT(*) FROM memories WHERE access_count = 0"
    ).fetchone()[0]
    never_accessed_pct = (zero_access / total * 100) if total > 0 else 0

    # Breakdown by event_type: avg access_count per type
    type_rows = db._conn.execute(
        """SELECT event_type, COUNT(*) as cnt,
                  AVG(access_count) as avg_access,
                  SUM(CASE WHEN access_count = 0 THEN 1 ELSE 0 END) as zero_cnt
           FROM memories
           GROUP BY event_type
           ORDER BY avg_access DESC"""
    ).fetchall()
    by_type = []
    for row in type_rows:
        by_type.append({
            "event_type": row[0] or "unknown",
            "count": row[1],
            "avg_access_count": round(row[2], 2),
            "zero_access_count": row[3],
            "zero_access_pct": round(row[3] / row[1] * 100, 1) if row[1] > 0 else 0,
        })

    # Top 10 most-accessed memories
    top_rows = db._conn.execute(
        """SELECT node_id, content, access_count, event_type
           FROM memories
           WHERE access_count > 0
           ORDER BY access_count DESC LIMIT 10"""
    ).fetchall()
    top_accessed = []
    for row in top_rows:
        top_accessed.append({
            "id": row[0],
            "content": row[1][:100],
            "access_count": row[2],
            "event_type": row[3] or "unknown",
        })

    # Overall average
    avg_row = db._conn.execute(
        "SELECT AVG(access_count) FROM memories"
    ).fetchone()
    avg_access = round(avg_row[0], 2) if avg_row[0] is not None else 0

    return {
        "total_memories": total,
        "zero_access_count": zero_access,
        "never_accessed_pct": round(never_accessed_pct, 1),
        "avg_access_count": avg_access,
        "by_type": by_type,
        "top_accessed": top_accessed,
    }


# ---------------------------------------------------------------------------
# Public API -- Weekly Knowledge Digest
# ---------------------------------------------------------------------------


def get_weekly_digest(days: int = 7) -> Dict[str, Any]:
    """Generate a weekly knowledge digest with stats, trends, and highlights.

    Returns dict with: summary, type_breakdown, top_topics, growth, highlights.
    """
    db = _get_store()
    now = datetime.now(timezone.utc)
    cutoff = (now - timedelta(days=days)).isoformat()

    total = db.node_count()

    # Count memories created in the period
    try:
        with db._lock:
            row = db._conn.execute(
                "SELECT COUNT(*) FROM memories WHERE created_at >= ?", (cutoff,)
            ).fetchone()
        period_count = row[0] if row else 0
    except Exception:
        period_count = 0

    # Type breakdown for the period
    type_breakdown: Dict[str, int] = {}
    try:
        with db._lock:
            rows = db._conn.execute(
                "SELECT event_type, COUNT(*) FROM memories "
                "WHERE created_at >= ? AND event_type IS NOT NULL "
                "GROUP BY event_type ORDER BY COUNT(*) DESC",
                (cutoff,),
            ).fetchall()
        type_breakdown = {r[0]: r[1] for r in rows if r[0]}
    except Exception:
        pass

    # Session count for the period
    session_count = 0
    try:
        with db._lock:
            row = db._conn.execute(
                "SELECT COUNT(DISTINCT session_id) FROM memories "
                "WHERE created_at >= ? AND session_id IS NOT NULL",
                (cutoff,),
            ).fetchone()
        session_count = row[0] if row else 0
    except Exception:
        pass

    # Top topics: extract most common words from recent content (simple TF)
    top_topics: list[str] = []
    try:
        with db._lock:
            rows = db._conn.execute(
                "SELECT content FROM memories WHERE created_at >= ? LIMIT 200",
                (cutoff,),
            ).fetchall()
        word_counts: Dict[str, int] = {}
        _stop_words = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would", "could",
            "should", "may", "might", "shall", "can", "to", "of", "in", "for",
            "on", "with", "at", "by", "from", "as", "into", "through", "during",
            "before", "after", "above", "below", "between", "out", "off", "over",
            "under", "again", "further", "then", "once", "and", "but", "or", "nor",
            "not", "so", "yet", "both", "each", "few", "more", "most", "other",
            "some", "such", "no", "only", "own", "same", "than", "too", "very",
            "just", "because", "if", "when", "while", "how", "what", "which",
            "who", "whom", "this", "that", "these", "those", "it", "its", "my",
            "your", "his", "her", "our", "their", "all", "any", "up", "about",
            "error", "memory", "session", "plan", "decision", "captured",
        }
        for (content,) in rows:
            words = re.findall(r'[a-zA-Z_]{4,}', content.lower())
            for w in words:
                if w not in _stop_words:
                    word_counts[w] = word_counts.get(w, 0) + 1
        top_topics = [w for w, _ in sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:8]]
    except Exception:
        pass

    # Growth comparison with previous period
    prev_cutoff = (now - timedelta(days=days * 2)).isoformat()
    prev_count = 0
    try:
        with db._lock:
            row = db._conn.execute(
                "SELECT COUNT(*) FROM memories WHERE created_at >= ? AND created_at < ?",
                (prev_cutoff, cutoff),
            ).fetchone()
        prev_count = row[0] if row else 0
    except Exception:
        pass
    growth_pct = ((period_count - prev_count) / max(prev_count, 1)) * 100 if prev_count > 0 else 0

    return {
        "period_days": days,
        "total_memories": total,
        "period_new": period_count,
        "session_count": session_count,
        "type_breakdown": type_breakdown,
        "top_topics": top_topics,
        "growth_pct": round(growth_pct, 1),
        "prev_period_count": prev_count,
    }


# ---------------------------------------------------------------------------
# Public API -- Activity Summary (CLI)
# ---------------------------------------------------------------------------


def get_activity_summary(days: int = 7) -> Dict[str, Any]:
    """Gather activity data for the CLI activity command.

    Returns: {sessions: [...], tasks: [...], insights: [...], claims: [...]}
    """
    result: Dict[str, Any] = {"sessions": [], "tasks": [], "insights": [], "claims": []}

    # Recent insights from timeline
    try:
        db = _get_store()
        data = db.get_timeline(days=days, limit_per_day=10)
        if data:
            for day in sorted(data.keys(), reverse=True):
                for m in data[day]:
                    etype = (m.metadata or {}).get("event_type", "memory")
                    preview = m.content[:120].replace("\n", " ")
                    result["insights"].append(
                        {
                            "type": etype,
                            "preview": preview,
                            "created_at": m.created_at.isoformat() if m.created_at else "",
                            "id": m.id[:12] if m.id else "",
                        }
                    )
            # Limit to 15 most recent across all days
            result["insights"] = result["insights"][:15]
    except Exception as e:
        logger.warning(f"Activity summary: insights failed: {e}")

    # Coordination data (sessions, tasks, claims)
    try:
        from omega.coordination import get_manager

        mgr = get_manager()

        # Sessions (active)
        try:
            sessions = mgr.list_sessions(auto_clean=False)
            for s in sessions:
                result["sessions"].append(
                    {
                        "session_id": s.get("session_id", "")[:16],
                        "project": s.get("project", ""),
                        "task": s.get("task", ""),
                        "started_at": s.get("started_at", ""),
                        "last_heartbeat": s.get("last_heartbeat", ""),
                        "status": s.get("status", ""),
                    }
                )
        except Exception as e:
            logger.warning(f"Activity summary: sessions failed: {e}")

        # Tasks (pending + in_progress)
        try:
            for st in ("pending", "in_progress"):
                tasks = mgr.list_tasks(status=st)
                for t in tasks:
                    result["tasks"].append(
                        {
                            "id": t.get("id", ""),
                            "title": t.get("title", ""),
                            "status": t.get("status", ""),
                            "progress": t.get("progress", 0),
                            "created_at": t.get("created_at", ""),
                        }
                    )
        except Exception as e:
            logger.warning(f"Activity summary: tasks failed: {e}")

        # File + branch claims across active sessions
        try:
            for s in sessions:
                sid = s.get("session_id", "")
                claims = mgr.get_session_claims(sid)
                for fp in claims.get("file_claims", []):
                    result["claims"].append({"type": "file", "path": fp, "session": sid[:16]})
                for br in claims.get("branch_claims", []):
                    result["claims"].append({"type": "branch", "path": br, "session": sid[:16]})
        except Exception as e:
            logger.warning(f"Activity summary: claims failed: {e}")

    except ImportError:
        logger.info("Coordination module not available for activity summary")
    except Exception as e:
        logger.warning(f"Activity summary: coordination failed: {e}")

    return result


# ---------------------------------------------------------------------------
# Reminders (experimental)
# ---------------------------------------------------------------------------

# Regex for parsing human-friendly durations: "1h", "30m", "2d", "1w", "1d12h", "2 hours"
_DURATION_RE = re.compile(
    r"(?:(\d+)\s*w(?:eeks?)?)?\s*"
    r"(?:(\d+)\s*d(?:ays?)?)?\s*"
    r"(?:(\d+)\s*h(?:ours?|rs?)?)?\s*"
    r"(?:(\d+)\s*m(?:in(?:utes?|s?)?)?)?",
)


def parse_duration(text: str) -> timedelta:
    """Parse a human-friendly duration string into a timedelta.

    Supported formats: "1h", "30m", "2d", "1w", "1d12h", "2 hours", "30 minutes".
    Raises ValueError on invalid or zero duration.
    """
    text = text.strip().lower()
    m = _DURATION_RE.fullmatch(text)
    if not m or not any(m.groups()):
        raise ValueError(f"Invalid duration: {text!r}. Use e.g. '1h', '30m', '2d', '1w', '1d12h'.")
    weeks = int(m.group(1) or 0)
    days = int(m.group(2) or 0)
    hours = int(m.group(3) or 0)
    minutes = int(m.group(4) or 0)
    td = timedelta(weeks=weeks, days=days, hours=hours, minutes=minutes)
    if td.total_seconds() <= 0:
        raise ValueError("Duration must be positive.")
    return td


def create_reminder(
    text: str,
    duration: str,
    context: Optional[str] = None,
    session_id: Optional[str] = None,
    project: Optional[str] = None,
) -> dict:
    """Create a time-based reminder.

    Stores directly via SQLiteStore.store() to bypass dedup/evolution —
    identical reminder text with different times should create separate entries.
    """
    td = parse_duration(duration)
    now = datetime.now(timezone.utc)
    remind_at = now + td

    meta = {
        "event_type": "reminder",
        "reminder_status": "pending",
        "remind_at": remind_at.isoformat(),
        "created_at_utc": now.isoformat(),
        "notified_out_of_session": False,
    }
    if context:
        meta["context"] = context
    if session_id:
        meta["session_id"] = session_id
    if project:
        meta["project"] = project

    # Include remind_at in content to avoid content-hash dedup
    # (same text at different times = different reminders)
    store_content = f"{text}\n[due: {remind_at.isoformat()}]"

    db = _get_store()
    node_id = db.store(
        content=store_content,
        session_id=session_id,
        metadata=meta,
        ttl_seconds=None,  # Permanent until dismissed
        skip_inference=True,  # Skip embedding dedup — same text, different times = different reminders
    )

    # Human-readable local time
    try:
        local_str = remind_at.astimezone().strftime("%Y-%m-%d %H:%M %Z")
    except Exception:
        local_str = remind_at.isoformat()

    return {
        "reminder_id": node_id,
        "text": text,
        "remind_at": remind_at.isoformat(),
        "remind_at_local": local_str,
        "duration": duration,
    }


def list_reminders(
    status: Optional[str] = None,
    include_dismissed: bool = False,
) -> List[Dict[str, Any]]:
    """List reminders, sorted by overdue first then by remind_at ascending."""
    db = _get_store()
    with db._lock:
        rows = db._conn.execute(
            "SELECT node_id, content, metadata, created_at FROM memories WHERE event_type = 'reminder'"
        ).fetchall()

    now = datetime.now(timezone.utc)
    # Regex to strip the internal [due: ...] suffix from stored content
    _due_suffix_re = re.compile(r"\n\[due: [^\]]+\]$")

    results = []
    for node_id, content, meta_json, created_at in rows:
        try:
            meta = json.loads(meta_json) if isinstance(meta_json, str) else (meta_json or {})
        except (json.JSONDecodeError, TypeError):
            meta = {}

        r_status = meta.get("reminder_status", "pending")

        # Filter by status
        if status and status != "all" and r_status != status:
            continue
        if not include_dismissed and not status and r_status == "dismissed":
            continue

        remind_at_str = meta.get("remind_at", "")
        try:
            remind_at = datetime.fromisoformat(remind_at_str)
            if remind_at.tzinfo is None:
                remind_at = remind_at.replace(tzinfo=timezone.utc)
        except (ValueError, TypeError):
            remind_at = now

        is_due = now >= remind_at
        is_overdue = is_due and r_status == "pending"
        time_until = remind_at - now

        try:
            remind_at_local = remind_at.astimezone().strftime("%Y-%m-%d %H:%M %Z")
        except Exception:
            remind_at_local = remind_at.isoformat()

        # Strip internal [due: ...] suffix for clean display
        clean_text = _due_suffix_re.sub("", content)

        results.append({
            "id": node_id,
            "text": clean_text,
            "status": r_status,
            "remind_at": remind_at.isoformat(),
            "remind_at_local": remind_at_local,
            "is_due": is_due,
            "is_overdue": is_overdue,
            "time_until": str(time_until).split(".")[0] if not is_due else "overdue",
            "context": meta.get("context"),
            "created_at": created_at,
        })

    # Sort: overdue first, then by remind_at ascending
    results.sort(key=lambda r: (not r["is_overdue"], r["remind_at"]))
    return results


def dismiss_reminder(reminder_id: str) -> Dict[str, Any]:
    """Dismiss a reminder by updating its status."""
    db = _get_store()
    node = db.get_node(reminder_id)
    if node is None:
        return {"success": False, "error": f"Reminder {reminder_id} not found"}

    meta = dict(node.metadata or {})
    if meta.get("event_type") != "reminder":
        return {"success": False, "error": f"{reminder_id} is not a reminder"}

    meta["reminder_status"] = "dismissed"
    meta["dismissed_at"] = datetime.now(timezone.utc).isoformat()
    db.update_node(reminder_id, metadata=meta)
    clean_text = re.sub(r"\n\[due: [^\]]+\]$", "", node.content)
    return {"success": True, "dismissed_id": reminder_id, "text": clean_text}


def get_due_reminders(mark_fired: bool = False) -> List[Dict[str, Any]]:
    """Get all pending reminders that are due now.

    If mark_fired=True, transitions their status from 'pending' to 'fired'.
    """
    all_reminders = list_reminders(status="pending")
    due = [r for r in all_reminders if r["is_due"]]

    if mark_fired and due:
        db = _get_store()
        now_iso = datetime.now(timezone.utc).isoformat()
        for r in due:
            node = db.get_node(r["id"])
            if node:
                meta = dict(node.metadata or {})
                meta["reminder_status"] = "fired"
                meta["fired_at"] = now_iso
                db.update_node(r["id"], metadata=meta)
                r["status"] = "fired"

    return due


# ---------------------------------------------------------------------------
# Module exports
# ---------------------------------------------------------------------------

__all__ = [
    "auto_capture",
    "query",
    "query_structured",
    "check_health",
    "get_dedup_stats",
    "export_memories",
    "import_memories",
    "welcome",
    "get_profile",
    "save_profile",
    "remember",
    "store",
    "delete_memory",
    "edit_memory",
    "extract_preferences",
    "list_preferences",
    "deduplicate",
    "reingest",
    "status",
    "get_cross_session_lessons",
    "get_cross_project_lessons",
    "reset_memory",
    "record_feedback",
    "clear_session",
    "batch_store",
    "find_similar_memories",
    "timeline",
    "consolidate",
    "traverse",
    "compact",
    "phrase_search",
    "type_stats",
    "session_stats",
    "check_constraints",
    "list_constraints",
    "save_constraints",
    "get_activity_summary",
    "get_weekly_digest",
    "parse_duration",
    "create_reminder",
    "list_reminders",
    "dismiss_reminder",
    "get_due_reminders",
]
