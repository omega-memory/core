"""
OMEGA SQLite Store -- SQLite-backed storage with sqlite-vec for vector search.

Replaces the in-memory graph system (OmegaMemory) with a single SQLite database.
All nodes, embeddings, and edges live on disk. Queries use SQL + vector similarity.

RAM impact: ~5-10 MB (SQLite overhead) vs 372 MB (in-memory graphs at 3,716 nodes).

Usage:
    store = SQLiteStore()
    node_id = store.store(content="Hello world", session_id="s1")
    results = store.query("hello", limit=5)
"""

import hashlib
import logging
import os
import re
import sqlite3
import struct
import threading
import unicodedata
import uuid
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from collections import OrderedDict

from omega import json_compat as json

import time as _time

logger = logging.getLogger("omega.sqlite_store")

SCHEMA_VERSION = 5
EMBEDDING_DIM = 384

# Pre-compiled regex for query deduplication (strip trailing git hashes)
_TRAILING_HASH_RE = re.compile(r"\s*-\s*[0-9a-f]{6,40}\s*$")

# Periodic TTL cleanup state
_last_cleanup: Optional[float] = None
_CLEANUP_INTERVAL = 3600  # seconds

# ---------------------------------------------------------------------------
# SQLite retry — handles multi-process write contention on shared omega.db.
# WAL mode + busy_timeout handle most cases, but under heavy contention
# (3+ MCP server processes) the busy_timeout can still expire. This wrapper
# retries with exponential backoff before surfacing the error.
# ---------------------------------------------------------------------------
_DB_RETRY_ATTEMPTS = 3
_DB_RETRY_BASE_DELAY = 1.0  # seconds


def _retry_on_locked(fn, *args, **kwargs):
    """Call fn with retry on 'database is locked' OperationalError."""
    for attempt in range(_DB_RETRY_ATTEMPTS):
        try:
            return fn(*args, **kwargs)
        except sqlite3.OperationalError as e:
            if "database is locked" in str(e) and attempt < _DB_RETRY_ATTEMPTS - 1:
                delay = _DB_RETRY_BASE_DELAY * (2 ** attempt)
                logger.warning("database is locked (attempt %d/%d), retrying in %.1fs",
                               attempt + 1, _DB_RETRY_ATTEMPTS, delay)
                _time.sleep(delay)
            else:
                raise


# ---------------------------------------------------------------------------
# Query result cache — avoids re-running vector + FTS5 pipeline for repeated
# queries within a short window (e.g., surface_memories on sequential edits).
# Invalidated on any write (store/delete/update).
# ---------------------------------------------------------------------------
_QUERY_CACHE_MAX = 128
_QUERY_CACHE_TTL_S = 60  # seconds
_QUERY_CACHE_WARM_TTL_S = 300  # seconds — extended TTL for high-confidence results (#2)
_HOT_CACHE_SIZE = 50  # Top N memories by access_count to keep in-memory (#2)
_HOT_CACHE_REFRESH_S = 300  # Refresh hot cache every 5 minutes (#2)
_TRIGRAM_FINGERPRINT_CHARS = 200  # Max chars for trigram fingerprint (#1)
_FAST_PATH_MIN_OVERLAP = 0.60  # Minimum trigram Jaccard for fast-path match (#1)

# Regex for content canonicalization (#6)
_MARKDOWN_STRIP_RE = re.compile(r'[*#`~\[\]()>|_]')
_WHITESPACE_COLLAPSE_RE = re.compile(r'\s+')


# ---------------------------------------------------------------------------
# Surfacing context (#4) — dynamic threshold profiles
# ---------------------------------------------------------------------------

class SurfacingContext(Enum):
    """Context in which memories are being surfaced."""
    GENERAL = "general"
    ERROR_DEBUG = "error_debug"
    FILE_EDIT = "file_edit"
    SESSION_START = "session_start"

# Thresholds per context: (min_vec_similarity, min_text_relevance, min_composite_score, context_weight_boost)
_SURFACING_THRESHOLDS = {
    SurfacingContext.GENERAL:       (0.50, 0.35, 0.10, 1.0),
    SurfacingContext.ERROR_DEBUG:   (0.40, 0.45, 0.08, 1.0),
    SurfacingContext.FILE_EDIT:     (0.50, 0.35, 0.10, 2.0),
    SurfacingContext.SESSION_START: (0.60, 0.45, 0.15, 1.0),
}


# ---------------------------------------------------------------------------
# Query intent (#5) — adaptive retrieval budget
# ---------------------------------------------------------------------------

class QueryIntent(Enum):
    """Classified intent for adaptive phase weighting."""
    FACTUAL = "factual"
    CONCEPTUAL = "conceptual"
    NAVIGATIONAL = "navigational"

# Intent weights: (vec, text, word_overlap, context, graph)
_INTENT_WEIGHTS = {
    QueryIntent.FACTUAL:      (0.3, 1.5, 1.8, 1.0, 1.0),
    QueryIntent.CONCEPTUAL:   (1.8, 0.5, 0.3, 1.0, 1.0),
    QueryIntent.NAVIGATIONAL: (0.1, 2.0, 2.0, 0.5, 0.3),
}


# ---------------------------------------------------------------------------
# Content canonicalization (#6)
# ---------------------------------------------------------------------------

def _canonicalize(text: str) -> str:
    """Canonicalize text for better matching: NFKC normalize, strip markdown, collapse whitespace."""
    text = unicodedata.normalize("NFKC", text)
    text = _MARKDOWN_STRIP_RE.sub(" ", text)
    text = _WHITESPACE_COLLAPSE_RE.sub(" ", text).strip()
    return text.lower()


def _trigram_fingerprint(text: str) -> frozenset:
    """Compute character-level trigram fingerprint for fast-path lookup (#1)."""
    canonical = _canonicalize(text[:_TRIGRAM_FINGERPRINT_CHARS])
    if len(canonical) < 3:
        return frozenset()
    return frozenset(canonical[i:i+3] for i in range(len(canonical) - 2))


def _trigram_jaccard(fp_a: frozenset, fp_b: frozenset) -> float:
    """Jaccard similarity between two trigram fingerprints."""
    if not fp_a or not fp_b:
        return 0.0
    intersection = len(fp_a & fp_b)
    union = len(fp_a | fp_b)
    return intersection / union if union > 0 else 0.0



def _serialize_f32(vector: List[float]) -> bytes:
    """Serialize a float32 vector to bytes for sqlite-vec."""
    return struct.pack(f"{len(vector)}f", *vector)


def _deserialize_f32(data: bytes, dim: int = EMBEDDING_DIM) -> List[float]:
    """Deserialize bytes to a float32 vector."""
    return list(struct.unpack(f"{dim}f", data))


# ---------------------------------------------------------------------------
# MemoryResult -- lightweight result object matching MemoryNode interface
# ---------------------------------------------------------------------------


class MemoryResult:
    """Lightweight result object that matches the MemoryNode interface used by bridge.py."""

    __slots__ = (
        "id",
        "content",
        "metadata",
        "created_at",
        "access_count",
        "last_accessed",
        "ttl_seconds",
        "relevance",
        "embedding",
        "_content_lower",
    )

    def __init__(
        self,
        id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        created_at: Optional[datetime] = None,
        access_count: int = 0,
        last_accessed: Optional[datetime] = None,
        ttl_seconds: Optional[int] = None,
        relevance: float = 0.0,
        embedding: Optional[List[float]] = None,
    ):
        self.id = id
        self.content = content
        self.metadata = metadata or {}
        self.created_at = created_at or datetime.now(timezone.utc)
        self.access_count = access_count
        self.last_accessed = last_accessed
        self.ttl_seconds = ttl_seconds
        self.relevance = relevance
        self.embedding = embedding
        self._content_lower = None

    @property
    def content_lower(self) -> str:
        if self._content_lower is None:
            self._content_lower = self.content.lower()
        return self._content_lower

    @property
    def expires_at(self) -> Optional[datetime]:
        if self.ttl_seconds is None:
            return None
        return self.created_at + timedelta(seconds=self.ttl_seconds)

    def is_expired(self, now: Optional[datetime] = None) -> bool:
        if self.ttl_seconds is None:
            return False
        now = now or datetime.now(timezone.utc)
        # Normalize both sides to be TZ-aware for safe comparison
        ca = self.created_at
        if ca.tzinfo is None:
            ca = ca.replace(tzinfo=timezone.utc)
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        return now > ca + timedelta(seconds=self.ttl_seconds)

    def touch(self) -> None:
        self.access_count += 1
        self.last_accessed = datetime.now(timezone.utc)

    def time_until_expiry(self) -> Optional[timedelta]:
        if self.ttl_seconds is None:
            return None
        now = datetime.now(timezone.utc)
        exp = self.expires_at
        # Normalize to TZ-aware
        if exp is not None and exp.tzinfo is None:
            exp = exp.replace(tzinfo=timezone.utc)
        remaining = exp - now
        return remaining if remaining.total_seconds() > 0 else timedelta(0)


# ---------------------------------------------------------------------------
# SQLiteStore
# ---------------------------------------------------------------------------


class SQLiteStore:
    """SQLite-backed memory store with sqlite-vec for vector search.

    Drop-in replacement for OmegaMemory in bridge.py. All data lives on disk
    in a single SQLite database file.
    """

    # Type weights for query scoring (same as OmegaMemory)
    _TYPE_WEIGHTS = {
        "checkpoint": 2.5,
        "reminder": 3.0,
        "decision": 2.0,
        "lesson_learned": 2.0,
        "error_pattern": 2.0,
        "user_preference": 2.0,
        "task_completion": 1.4,
        "reflexion": 1.3,
        "outcome_evaluation": 1.3,
        "self_reflection": 1.3,
        "session_summary": 1.2,
        "preference_generated": 1.1,
        "advisor_action_outcome": 1.1,
        "sota_research": 1.4,
        "research_report": 1.3,
        "benchmark_update": 1.3,
        "sota_scan": 1.1,
        "file_conflict": 1.0,
        "merge_claim": 0.8,
        "merge_release": 0.8,
        "file_claimed": 0.7,
        "file_released": 0.7,
        "branch_claimed": 0.7,
        "branch_released": 0.7,
        "session_respawn": 0.5,
        "coordination_snapshot": 0.2,
        "test": 0.4,
        "code_chunk": 0.1,
        "file_summary": 0.05,
    }

    # Default priority per event type (1=lowest, 5=highest)
    _DEFAULT_PRIORITY = {
        "checkpoint": 5,
        "reminder": 5,
        "user_preference": 5,
        "error_pattern": 4,
        "lesson_learned": 4,
        "decision": 4,
        "task_completion": 3,
        "reflexion": 3,
        "outcome_evaluation": 3,
        "self_reflection": 3,
        "sota_research": 3,
        "research_report": 3,
        "session_summary": 2,
        "coordination_snapshot": 1,
        "session_respawn": 1,
        "file_summary": 1,
        "code_chunk": 1,
    }

    # Abstention thresholds — minimum quality for results to survive
    _MIN_VEC_SIMILARITY = 0.50  # Minimum cosine similarity for vec results
    _MIN_TEXT_RELEVANCE = 0.35  # Minimum raw word overlap ratio for text-only results
    _MIN_COMPOSITE_SCORE = 0.10  # Absolute floor on composite score (catches temporal penalty)
    _MIN_VEC_CANDIDATES = 20  # Floor on vector candidate pool (prevents small limit from dropping good matches)

    # Per-event-type retrieval profiles (ALMA-inspired).
    # Reweight scoring phases based on what works best for each memory type.
    # Tuple order: (vec, text, word_overlap, context, graph)
    _RETRIEVAL_PROFILES = {
        # --- Event-type profiles (production MCP queries) ---
        "error_pattern":    (0.3, 1.5, 2.0, 0.5, 0.3),  # Stack traces need keyword match
        "decision":         (0.8, 0.6, 0.5, 1.0, 2.0),  # Decisions chain to prior decisions
        "lesson_learned":   (1.5, 0.8, 0.5, 0.8, 1.0),  # Abstract knowledge = semantic
        "user_preference":  (0.6, 1.0, 1.5, 0.3, 0.3),  # Keyword + preference boost
        # --- Question-type retrieval profiles ---
        "single-session-assistant":  (1.0, 1.0, 1.0, 1.0, 1.0),
        "single-session-user":       (1.0, 1.1, 1.2, 1.0, 1.0),
        "knowledge-update":          (0.8, 1.3, 1.5, 1.0, 1.0),
        "single-session-preference": (0.8, 1.0, 1.5, 1.0, 1.0),
        "multi-session":             (1.3, 1.0, 1.3, 1.0, 1.0),
        "temporal-reasoning":        (1.0, 1.3, 1.3, 1.0, 1.0),
        # --- Fallback ---
        "_default":         (1.0, 1.0, 1.0, 1.0, 1.0),  # Preserves current behavior
    }

    _INFRASTRUCTURE_TYPES = frozenset(
        {
            "file_summary",
            "code_chunk",
            "session_respawn",
            "coordination_snapshot",
            "session_summary",  # exclude from user-facing queries
        }
    )

    DEFAULT_EMBEDDING_DEDUP_THRESHOLD = 0.88
    DEFAULT_JACCARD_DEDUP_THRESHOLD = 0.80

    # Input size limits (configurable via env vars)
    _MAX_NODES = int(os.environ.get("OMEGA_MAX_NODES", "50000"))
    _MAX_CONTENT_SIZE = int(os.environ.get("OMEGA_MAX_CONTENT_SIZE", "1000000"))  # 1MB

    def __init__(self, db_path=None):
        omega_home = Path(os.environ.get("OMEGA_HOME", str(Path.home() / ".omega")))
        self.db_path = Path(db_path) if db_path else (omega_home / "omega.db")
        self.db_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)

        self._lock = threading.Lock()
        self._vec_available = False
        self._query_cache: OrderedDict = OrderedDict()  # key → (timestamp, results)
        self._conn = self._connect()
        self._init_schema()

        # Merged retrieval profiles: built-in + plugin overrides
        self._retrieval_profiles_merged: Dict[str, tuple] = dict(self._RETRIEVAL_PROFILES)
        # Plugin score modifiers: list of fn(node_id, score, metadata) -> score
        self._score_modifiers: list = []
        # A/B feedback tracking: LRU cache of recent query contexts per memory
        self._recent_query_context: OrderedDict = OrderedDict()  # node_id → {query_text, query_hint, score, vec_sim, ts}
        _QUERY_CONTEXT_MAX = 50

        # Stats dict for bridge.py compatibility
        self.stats: Dict[str, Any] = {
            "stores": 0,
            "queries": 0,
            "hits": 0,
            "misses": 0,
            "auto_evictions": 0,
            "content_dedup_skips": 0,
            "memory_evolutions": 0,
            "embedding_dedup_skips": 0,
        }

        # Load persisted stats
        self._load_stats()

        # Engram-inspired caches (#2)
        self._hot_memories: Dict[str, MemoryResult] = {}
        self._hot_cache_ts: float = 0.0
        self._session_cache: Dict[str, List[MemoryResult]] = {}
        self._prefetch_cache: Dict[str, List[MemoryResult]] = {}
        self._refresh_hot_cache()

    def register_plugin_profiles(self, profiles: Dict[str, tuple]) -> None:
        """Register retrieval profiles from a plugin. Plugin profiles override
        built-in defaults for the same event_type key."""
        for key, weights in profiles.items():
            if isinstance(weights, (tuple, list)) and len(weights) == 5:
                self._retrieval_profiles_merged[key] = tuple(weights)
            else:
                logger.warning("Plugin profile %s has invalid shape, skipping", key)

    def register_score_modifier(self, modifier) -> None:
        """Register a plugin score modifier: fn(node_id, score, metadata) -> score."""
        self._score_modifiers.append(modifier)

    def _connect(self) -> sqlite3.Connection:
        """Create a new SQLite connection with optimal settings."""
        from omega.crypto import secure_connect

        conn = secure_connect(
            self.db_path,
            timeout=30,
            check_same_thread=False,
        )
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA cache_size=-16000")  # 16MB cache
        conn.execute("PRAGMA mmap_size=33554432")  # 32MB memory-mapped I/O
        conn.execute("PRAGMA busy_timeout=30000")  # 30s — handles multi-process contention
        conn.execute("PRAGMA foreign_keys=ON")

        # Try to load sqlite-vec extension
        try:
            import sqlite_vec

            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
            conn.enable_load_extension(False)
            self._vec_available = True
        except (ImportError, Exception) as e:
            logger.warning(f"sqlite-vec not available, falling back to brute-force: {e}")
            self._vec_available = False

        return conn

    def _init_schema(self) -> None:
        """Create tables if they don't exist."""
        c = self._conn

        c.execute("""
            CREATE TABLE IF NOT EXISTS schema_version (
                version INTEGER NOT NULL
            )
        """)

        # Check current version
        row = c.execute("SELECT version FROM schema_version LIMIT 1").fetchone()
        if row is None:
            c.execute("INSERT INTO schema_version (version) VALUES (?)", (SCHEMA_VERSION,))

        # Schema migration v1 → v2: add priority and referenced_date columns
        if row and row[0] < 2:
            try:
                c.execute("ALTER TABLE memories ADD COLUMN priority INTEGER DEFAULT 3")
            except Exception:
                pass  # Column already exists
            try:
                c.execute("ALTER TABLE memories ADD COLUMN referenced_date TEXT")
            except Exception:
                pass  # Column already exists
            c.execute("CREATE INDEX IF NOT EXISTS idx_memories_priority ON memories(priority)")
            c.execute("CREATE INDEX IF NOT EXISTS idx_memories_referenced_date ON memories(referenced_date)")
            c.execute("UPDATE schema_version SET version = 2")
            c.commit()
            logger.info("Schema migrated v1 → v2: added priority, referenced_date columns")

        # Schema migration v2 → v3: add entity_id column
        current_version = c.execute("SELECT version FROM schema_version LIMIT 1").fetchone()
        if current_version and current_version[0] < 3:
            try:
                c.execute("ALTER TABLE memories ADD COLUMN entity_id TEXT")
            except Exception:
                pass  # Column already exists
            c.execute("CREATE INDEX IF NOT EXISTS idx_memories_entity_id ON memories(entity_id)")
            c.execute("UPDATE schema_version SET version = 3")
            c.commit()
            logger.info("Schema migrated v2 → v3: added entity_id column")

        # Schema migration v3 → v4: add agent_type column
        current_version = c.execute("SELECT version FROM schema_version LIMIT 1").fetchone()
        if current_version and current_version[0] < 4:
            try:
                c.execute("ALTER TABLE memories ADD COLUMN agent_type TEXT")
            except Exception:
                pass  # Column already exists
            c.execute("CREATE INDEX IF NOT EXISTS idx_memories_agent_type ON memories(agent_type)")
            c.execute("UPDATE schema_version SET version = 4")
            c.commit()
            logger.info("Schema migrated v3 → v4: added agent_type column")

        # Schema migration v4 → v5: add canonical_hash column (#6 Engram)
        current_version = c.execute("SELECT version FROM schema_version LIMIT 1").fetchone()
        if current_version and current_version[0] < 5:
            try:
                c.execute("ALTER TABLE memories ADD COLUMN canonical_hash TEXT")
            except Exception:
                pass  # Column already exists
            c.execute("CREATE INDEX IF NOT EXISTS idx_memories_canonical_hash ON memories(canonical_hash)")
            c.execute("UPDATE schema_version SET version = 5")
            c.commit()
            logger.info("Schema migrated v4 → v5: added canonical_hash column")

        c.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                node_id TEXT UNIQUE NOT NULL,
                content TEXT NOT NULL,
                metadata TEXT,
                created_at TEXT NOT NULL,
                last_accessed TEXT,
                access_count INTEGER DEFAULT 0,
                ttl_seconds INTEGER,
                session_id TEXT,
                event_type TEXT,
                project TEXT,
                content_hash TEXT,
                priority INTEGER DEFAULT 3,
                referenced_date TEXT,
                entity_id TEXT,
                agent_type TEXT,
                canonical_hash TEXT
            )
        """)

        # Indexes
        for col in (
            "node_id",
            "event_type",
            "session_id",
            "project",
            "created_at",
            "content_hash",
            "priority",
            "referenced_date",
            "entity_id",
            "agent_type",
            "canonical_hash",
        ):
            c.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_memories_{col}
                ON memories({col})
            """)

        # sqlite-vec virtual table
        if self._vec_available:
            try:
                c.execute(f"""
                    CREATE VIRTUAL TABLE IF NOT EXISTS memories_vec
                    USING vec0(embedding float[{EMBEDDING_DIM}] distance_metric=cosine)
                """)
            except Exception as e:
                logger.warning(f"Failed to create vec table: {e}")
                self._vec_available = False

        # FTS5 full-text search index
        try:
            c.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts
                USING fts5(content, content='memories', content_rowid='id')
            """)
            self._fts_available = True
        except Exception as e:
            logger.debug(f"FTS5 not available: {e}")
            self._fts_available = False

        # Edges table (temporal, causal)
        c.execute("""
            CREATE TABLE IF NOT EXISTS edges (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                edge_type TEXT NOT NULL,
                weight REAL DEFAULT 1.0,
                metadata TEXT,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                UNIQUE(source_id, target_id, edge_type)
            )
        """)
        for col in ("source_id", "target_id", "edge_type"):
            c.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_edges_{col}
                ON edges({col})
            """)

        # Drop dead entity_index table (was never used)
        c.execute("DROP TABLE IF EXISTS entity_index")

        # FTS5 sync triggers (keep FTS index in sync with memories table)
        if self._fts_available:
            try:
                c.execute("""
                    CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
                        INSERT INTO memories_fts(rowid, content) VALUES (new.id, new.content);
                    END
                """)
                c.execute("""
                    CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
                        INSERT INTO memories_fts(memories_fts, rowid, content) VALUES ('delete', old.id, old.content);
                    END
                """)
                c.execute("""
                    CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE OF content ON memories BEGIN
                        INSERT INTO memories_fts(memories_fts, rowid, content) VALUES ('delete', old.id, old.content);
                        INSERT INTO memories_fts(rowid, content) VALUES (new.id, new.content);
                    END
                """)
                # Populate FTS from existing data if empty
                fts_count = c.execute("SELECT COUNT(*) FROM memories_fts").fetchone()[0]
                mem_count = c.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
                if fts_count == 0 and mem_count > 0:
                    c.execute("INSERT INTO memories_fts(rowid, content) SELECT id, content FROM memories")
                    logger.info(f"Populated FTS5 index with {mem_count} existing memories")
            except Exception as e:
                logger.debug(f"FTS5 trigger setup failed: {e}")
                self._fts_available = False

        c.commit()

    # ------------------------------------------------------------------
    # Resilient commit — retries on multi-process lock contention
    # ------------------------------------------------------------------

    def _commit(self) -> None:
        """Commit with retry on 'database is locked'.

        WAL mode + busy_timeout=30s handles most contention, but under
        heavy multi-process load (3+ MCP servers) the timeout can still
        expire.  This retries with exponential backoff before giving up.
        """
        _retry_on_locked(self._conn.commit)

    # ------------------------------------------------------------------
    # Core CRUD
    # ------------------------------------------------------------------

    def _invalidate_query_cache(self) -> None:
        """Clear query result cache after writes."""
        self._query_cache.clear()
        self._hot_cache_ts = 0.0  # Force hot cache refresh on next query (#2)

    def store(
        self,
        content: str,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        embedding: Optional[List[float]] = None,
        dependencies: Optional[List[str]] = None,
        ttl_seconds: Optional[int] = None,
        graphs: Optional[List[str]] = None,
        skip_inference: bool = False,
        entity_id: Optional[str] = None,
        agent_type: Optional[str] = None,
    ) -> str:
        """Store a memory. Returns the node ID."""
        if not content:
            raise ValueError("content must be a non-empty string")
        if len(content) > self._MAX_CONTENT_SIZE:
            raise ValueError(
                f"Content size ({len(content):,} bytes) exceeds limit ({self._MAX_CONTENT_SIZE:,} bytes). "
                "Override with OMEGA_MAX_CONTENT_SIZE env var."
            )
        if self._MAX_NODES > 0:
            count = self._conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
            if count >= self._MAX_NODES:
                raise ValueError(
                    f"Node count ({count:,}) has reached the limit ({self._MAX_NODES:,}). "
                    "Run omega_consolidate to prune, or raise OMEGA_MAX_NODES env var."
                )
        self._invalidate_query_cache()
        meta = dict(metadata or {})
        if session_id:
            meta["session_id"] = session_id

        # Auto-generate embedding if not provided (outside lock — CPU-bound)
        if embedding is None:
            from omega.graphs import generate_embedding, get_embedding_model_info, get_active_backend

            embedding = generate_embedding(content)
            # If we fell back to hash, don't store the embedding (would corrupt vec search)
            if get_active_backend() is None:
                logger.warning("store: hash-fallback embedding discarded — text search only")
                embedding = None
            try:
                model_info = get_embedding_model_info()
                meta["_embedding_model"] = model_info["model_name"]
                meta["_embedding_model_version"] = model_info["model_version"]
            except Exception as e:
                logger.debug("Could not attach embedding model info: %s", e)

        content_hash = hashlib.sha256(content.encode()).hexdigest()
        canonical_hash = hashlib.sha256(_canonicalize(content).encode()).hexdigest()

        with self._lock:
            # Canonical dedup (#6): catch reformatted duplicates
            canonical_existing = self._conn.execute(
                """SELECT node_id, id FROM memories WHERE canonical_hash = ?
                   AND (ttl_seconds IS NULL
                        OR datetime(created_at, '+' || ttl_seconds || ' seconds') > datetime('now'))
                   LIMIT 1""",
                (canonical_hash,),
            ).fetchone()
            if canonical_existing:
                self._conn.execute(
                    "UPDATE memories SET access_count = access_count + 1 WHERE node_id = ?",
                    (canonical_existing[0],),
                )
                self._commit()
                self.stats.setdefault("dedup_canonical", 0)
                self.stats["dedup_canonical"] += 1
                return canonical_existing[0]

            # Exact-match dedup via content hash (skip expired memories)
            existing = self._conn.execute(
                """SELECT node_id, id FROM memories WHERE content_hash = ?
                   AND (ttl_seconds IS NULL
                        OR datetime(created_at, '+' || ttl_seconds || ' seconds') > datetime('now'))
                   LIMIT 1""",
                (content_hash,),
            ).fetchone()
            if existing:
                self._conn.execute(
                    "UPDATE memories SET access_count = access_count + 1 WHERE node_id = ?", (existing[0],)
                )
                self._commit()
                self.stats.setdefault("dedup_exact", 0)
                self.stats["dedup_exact"] += 1
                return existing[0]

            # Embedding-based dedup
            if embedding and not skip_inference and self._vec_available:
                try:
                    similar = self._vec_query(embedding, limit=1)
                    if similar:
                        top_rowid, distance = similar[0]
                        similarity = 1.0 - distance  # cosine distance -> similarity
                        if similarity >= self.DEFAULT_EMBEDDING_DEDUP_THRESHOLD:
                            row = self._conn.execute(
                                "SELECT node_id FROM memories WHERE id = ?", (top_rowid,)
                            ).fetchone()
                            if row:
                                self._conn.execute(
                                    "UPDATE memories SET access_count = access_count + 1 WHERE id = ?", (top_rowid,)
                                )
                                self._commit()
                                self.stats.setdefault("dedup_skips", 0)
                                self.stats["dedup_skips"] += 1
                                return row[0]
                except Exception as e:
                    logger.debug(f"Embedding dedup check failed: {e}")

            # Generate node ID
            node_id = f"mem-{uuid.uuid4().hex[:12]}"

            event_type = meta.get("event_type") or meta.get("type")
            project = meta.get("project")
            now = datetime.now(timezone.utc).isoformat()

            # Determine priority from metadata or event type default
            priority = meta.get("priority") or self._DEFAULT_PRIORITY.get(event_type, 3)
            referenced_date = meta.get("referenced_date")

            # Wire entity_id from metadata if not passed directly
            effective_entity_id = entity_id or meta.get("entity_id")

            # Wire agent_type from metadata if not passed directly
            effective_agent_type = agent_type or meta.get("agent_type")

            self._conn.execute(
                """INSERT INTO memories
                   (node_id, content, metadata, created_at, access_count,
                    ttl_seconds, session_id, event_type, project, content_hash,
                    priority, referenced_date, entity_id, agent_type, canonical_hash)
                   VALUES (?, ?, ?, ?, 0, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    node_id,
                    content,
                    json.dumps(meta),
                    now,
                    ttl_seconds,
                    session_id,
                    event_type,
                    project,
                    content_hash,
                    priority,
                    referenced_date,
                    effective_entity_id,
                    effective_agent_type,
                    canonical_hash,
                ),
            )

            # Get the rowid for the vec table
            rowid = self._conn.execute("SELECT id FROM memories WHERE node_id = ?", (node_id,)).fetchone()[0]

            # Insert embedding into vec table
            if embedding and self._vec_available:
                try:
                    self._conn.execute(
                        "INSERT INTO memories_vec (rowid, embedding) VALUES (?, ?)", (rowid, _serialize_f32(embedding))
                    )
                except Exception as e:
                    logger.debug(f"Vec insert failed: {e}")

            # Add causal edges if dependencies provided
            if dependencies:
                for dep_id in dependencies:
                    self._conn.execute(
                        """INSERT INTO edges (source_id, target_id, edge_type, created_at)
                           VALUES (?, ?, 'causal', ?)""",
                        (node_id, dep_id, now),
                    )

            self._commit()
            self.stats["stores"] += 1

        return node_id

    def get_node(self, node_id: str) -> Optional[MemoryResult]:
        """Get a node by ID. Updates access tracking."""
        with self._lock:
            row = self._conn.execute(
                """SELECT node_id, content, metadata, created_at, access_count,
                          last_accessed, ttl_seconds
                   FROM memories WHERE node_id = ?""",
                (node_id,),
            ).fetchone()
            if not row:
                return None

            # Update access tracking
            now = datetime.now(timezone.utc).isoformat()
            self._conn.execute(
                "UPDATE memories SET access_count = access_count + 1, last_accessed = ? WHERE node_id = ?",
                (now, node_id),
            )
            self._commit()

            return self._row_to_result(row)

    def delete_node(self, node_id: str) -> bool:
        """Delete a node and its edges."""
        self._invalidate_query_cache()
        with self._lock:
            # Get rowid for vec table cleanup
            row = self._conn.execute("SELECT id FROM memories WHERE node_id = ?", (node_id,)).fetchone()
            if not row:
                return False

            rowid = row[0]

            self._conn.execute("DELETE FROM memories WHERE node_id = ?", (node_id,))
            self._conn.execute("DELETE FROM edges WHERE source_id = ? OR target_id = ?", (node_id, node_id))

            if self._vec_available:
                try:
                    self._conn.execute("DELETE FROM memories_vec WHERE rowid = ?", (rowid,))
                except Exception as e:
                    logger.debug("Failed to delete vec embedding rowid=%s: %s", rowid, e)

            self._commit()
        return True

    def node_count(self) -> int:
        """Return total number of memories."""
        row = self._conn.execute("SELECT COUNT(*) FROM memories").fetchone()
        return row[0] if row else 0

    def edge_count(self) -> int:
        """Return total number of edges."""
        row = self._conn.execute("SELECT COUNT(*) FROM edges").fetchone()
        return row[0] if row else 0

    def get_last_capture_time(self) -> Optional[str]:
        """Return ISO timestamp of the most recent memory, or None."""
        row = self._conn.execute("SELECT created_at FROM memories ORDER BY created_at DESC LIMIT 1").fetchone()
        return row[0] if row else None

    def get_session_event_counts(self, session_id: str) -> Dict[str, int]:
        """Count memories by event_type for a given session."""
        rows = self._conn.execute(
            "SELECT event_type, COUNT(*) "
            "FROM memories WHERE session_id = ? AND event_type IS NOT NULL "
            "GROUP BY event_type",
            (session_id,),
        ).fetchall()
        return {r[0]: r[1] for r in rows if r[0]}

    def update_node(
        self,
        node_id: str,
        content: Optional[str] = None,
        metadata: Optional[Dict] = None,
        access_count: Optional[int] = None,
    ) -> bool:
        """Update fields on an existing node."""
        self._invalidate_query_cache()
        sets = []
        params = []
        new_embedding = None
        if content is not None:
            sets.append("content = ?")
            params.append(content)
            sets.append("content_hash = ?")
            params.append(hashlib.sha256(content.encode()).hexdigest())
            sets.append("canonical_hash = ?")
            params.append(hashlib.sha256(_canonicalize(content).encode()).hexdigest())
            # Re-embed to keep vec table in sync (CPU-bound, done outside lock)
            if self._vec_available:
                try:
                    from omega.graphs import generate_embedding, get_active_backend

                    new_embedding = generate_embedding(content)
                    if get_active_backend() is None:
                        new_embedding = None  # Hash fallback — don't store
                except Exception as e:
                    logger.debug("update_node: re-embed failed: %s", e)
        if metadata is not None:
            sets.append("metadata = ?")
            params.append(json.dumps(metadata))
            # Update denormalized columns
            sets.append("event_type = ?")
            params.append(metadata.get("event_type") or metadata.get("type"))
            sets.append("session_id = ?")
            params.append(metadata.get("session_id"))
            sets.append("project = ?")
            params.append(metadata.get("project"))
        if access_count is not None:
            sets.append("access_count = ?")
            params.append(access_count)

        if not sets:
            return False

        with self._lock:
            params.append(node_id)
            self._conn.execute(f"UPDATE memories SET {', '.join(sets)} WHERE node_id = ?", params)
            # Update vec embedding if content changed
            if new_embedding is not None:
                row = self._conn.execute("SELECT id FROM memories WHERE node_id = ?", (node_id,)).fetchone()
                if row:
                    try:
                        self._conn.execute("DELETE FROM memories_vec WHERE rowid = ?", (row[0],))
                        self._conn.execute(
                            "INSERT INTO memories_vec (rowid, embedding) VALUES (?, ?)",
                            (row[0], _serialize_f32(new_embedding)),
                        )
                    except Exception as e:
                        logger.debug("update_node: vec update failed: %s", e)
            self._commit()
        return True

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(
        self,
        query_text: str,
        limit: int = 10,
        session_id: Optional[str] = None,
        use_cache: bool = True,
        expand_query: bool = True,
        exclude_types: Optional[List[str]] = None,
        include_infrastructure: bool = False,
        project_path: str = "",
        scope: str = "project",
        context_file: str = "",
        context_tags: Optional[List[str]] = None,
        temporal_range: Optional[tuple] = None,
        entity_id: Optional[str] = None,
        agent_type: Optional[str] = None,
        query_hint: Optional[str] = None,
        surfacing_context: Optional[SurfacingContext] = None,
    ) -> List[MemoryResult]:
        """Search memories using vector similarity + text matching.

        When context_file or context_tags are provided, results whose tags,
        project, or file paths overlap with the current context receive a
        relevance boost, improving results for the user's active work.

        When use_cache is True (default), identical queries within the TTL
        window return cached results, avoiding the full vector+FTS5 pipeline.

        surfacing_context controls dynamic threshold profiles (#4 Engram).
        """
        global _last_cleanup
        now_mono = _time.monotonic()

        # Resolve surfacing context thresholds (#4)
        _ctx = surfacing_context or SurfacingContext.GENERAL
        _ctx_thresholds = _SURFACING_THRESHOLDS.get(_ctx, _SURFACING_THRESHOLDS[SurfacingContext.GENERAL])
        ctx_min_vec, ctx_min_text, ctx_min_composite, ctx_weight_boost = _ctx_thresholds

        # --- Query result cache: check (tiered TTL #2) ---
        _cache_key = None
        if use_cache:
            _cache_key = (
                query_text, limit, session_id,
                tuple(sorted(exclude_types)) if exclude_types else (),
                include_infrastructure, project_path, scope,
                context_file, tuple(context_tags) if context_tags else (),
                temporal_range, entity_id, agent_type, query_hint,
                surfacing_context,
            )
            cached = self._query_cache.get(_cache_key)
            if cached is not None:
                ts, results, confidence = cached
                ttl = _QUERY_CACHE_WARM_TTL_S if confidence > 0.7 else _QUERY_CACHE_TTL_S
                if (now_mono - ts) < ttl:
                    self._query_cache.move_to_end(_cache_key)
                    self.stats["queries"] += 1
                    self.stats["hits"] += 1
                    return results
                else:
                    del self._query_cache[_cache_key]
        if _last_cleanup is None or (now_mono - _last_cleanup) > _CLEANUP_INTERVAL:
            _last_cleanup = now_mono
            try:
                self.cleanup_expired()
            except Exception as e:
                logger.warning(f"Periodic cleanup failed: {e}")

        self.stats["queries"] += 1

        if (now_mono - self._hot_cache_ts) > _HOT_CACHE_REFRESH_S:
            self._refresh_hot_cache()
        fast_path_results = self._fast_path_lookup(query_text, limit=limit)
        if fast_path_results:
            self.stats["fast_path_hits"] = self.stats.get("fast_path_hits", 0) + 1
            if _cache_key is not None:
                self._query_cache[_cache_key] = (now_mono, fast_path_results, 0.9)
                while len(self._query_cache) > _QUERY_CACHE_MAX:
                    self._query_cache.popitem(last=False)
            if session_id:
                self._session_cache[session_id] = fast_path_results
            return fast_path_results
        hot_results = self._check_hot_tier(query_text, limit=limit)
        if hot_results:
            self.stats["hot_cache_hits"] = self.stats.get("hot_cache_hits", 0) + 1
        query_intent = self._classify_query_intent(query_text)

        # Resolve retrieval profile for phase weighting (ALMA-inspired)
        _profile = self._retrieval_profiles_merged.get(
            query_hint, self._retrieval_profiles_merged.get("_default", (1.0, 1.0, 1.0, 1.0, 1.0))
        ) if query_hint else self._retrieval_profiles_merged.get("_default", (1.0, 1.0, 1.0, 1.0, 1.0))
        pw_vec, pw_text, pw_word, pw_ctx, pw_graph = _profile

        # Apply adaptive intent-based weights (#3)
        if query_intent and not query_hint:
            iw = _INTENT_WEIGHTS.get(query_intent, (1.0, 1.0, 1.0, 1.0, 1.0))
            pw_vec *= iw[0]
            pw_text *= iw[1]
            pw_word *= iw[2]
            pw_ctx *= iw[3]
            pw_graph *= iw[4]

        # Apply context weight boost (#4)
        pw_ctx *= ctx_weight_boost

        all_results: Dict[str, MemoryResult] = {}
        node_scores: Dict[str, float] = {}
        raw_vec_sims: Dict[str, float] = {}

        # Seed with hot cache results (#2)
        for hr in hot_results:
            all_results[hr.id] = hr
            node_scores[hr.id] = hr.relevance * 0.8

        # Keyword pre-filter: skip embedding for keyword-driven queries
        skip_vec = self._is_keyword_sufficient(query_text) or query_intent == QueryIntent.NAVIGATIONAL
        if skip_vec:
            self.stats["vec_skips"] = self.stats.get("vec_skips", 0) + 1

        # Phase 1: Vector similarity search
        if self._vec_available and not skip_vec:
            try:
                from omega.graphs import generate_embedding

                query_emb = generate_embedding(query_text)
                if query_emb:
                    vec_mult = 5
                    vec_limit = max(limit * vec_mult, self._MIN_VEC_CANDIDATES)
                    vec_results = self._vec_query(query_emb, limit=vec_limit)
                    for rowid, distance in vec_results:
                        row = self._conn.execute(
                            """SELECT node_id, content, metadata, created_at,
                                      access_count, last_accessed, ttl_seconds
                               FROM memories WHERE id = ?""",
                            (rowid,),
                        ).fetchone()
                        if row:
                            result = self._row_to_result(row)
                            similarity = 1.0 - distance
                            if similarity < 0.1:
                                continue
                            event_type = result.metadata.get("event_type", "")
                            type_weight = self._TYPE_WEIGHTS.get(event_type, 1.0)
                            fb_score = result.metadata.get("feedback_score", 0)
                            fb_factor = self._compute_fb_factor(fb_score)
                            priority = result.metadata.get("priority", 3)
                            priority_factor = 0.7 + (priority * 0.08)  # 0.78 (pri=1) to 1.10 (pri=5)
                            score = similarity * type_weight * fb_factor * priority_factor * pw_vec
                            # Consolidation quality boost (compacted knowledge nodes)
                            cq = result.metadata.get("consolidation_quality", 0)
                            if cq > 0:
                                score *= 1.0 + min(cq, 3.0) * 0.1  # up to 1.3x
                            result.relevance = similarity
                            raw_vec_sims[result.id] = similarity
                            all_results[result.id] = result
                            node_scores[result.id] = score
            except Exception as e:
                logger.debug(f"Vector search failed: {e}")

        # Phase 2: Text-based fallback/supplement
        text_mult = 4 if temporal_range else 3
        text_results = self._text_search(query_text, limit=limit * text_mult)
        for result in text_results:
            if result.id not in all_results:
                event_type = result.metadata.get("event_type", "")
                type_weight = self._TYPE_WEIGHTS.get(event_type, 1.0)
                fb_score = result.metadata.get("feedback_score", 0)
                fb_factor = self._compute_fb_factor(fb_score)
                priority = result.metadata.get("priority", 3)
                priority_factor = 0.7 + (priority * 0.08)
                score = result.relevance * type_weight * fb_factor * priority_factor * pw_text
                all_results[result.id] = result
                node_scores[result.id] = score
            else:
                # Multiplicative boost for dual-match (found by both vec + text)
                text_rel = result.relevance
                node_scores[result.id] *= 1.3 + text_rel * 0.5

        # Phase 2.5: Word/tag overlap boost — rewards term matches between
        # query and content+tags, helping memories with precise keyword overlap
        # outrank semantically-similar but off-topic results.
        _query_words = [w for w in query_text.lower().split() if len(w) > 2]
        if _query_words:
            for nid in list(node_scores.keys()):
                node = all_results[nid]
                content_lower = node.content.lower()
                tag_text = " ".join(str(t).lower() for t in (node.metadata.get("tags") or []))
                searchable = content_lower + " " + tag_text
                word_ratio = self._word_overlap(_query_words, searchable)
                if word_ratio > 0:
                    # Dampen boost for negatively-rated memories so outdated
                    # facts can't use word overlap to outrank updated versions
                    fb = node.metadata.get("feedback_score", 0)
                    fb_mod = 0.5 if fb < 0 else 1.0
                    node_scores[nid] *= 1.0 + word_ratio * 0.5 * fb_mod * pw_word

        # Phase 2.6: Preference signal boost — when query contains preference
        # indicators, boost user_preference memories to surface them above noise.
        _PREFERENCE_SIGNALS = {
            "prefer", "preference", "favorite", "favourite", "like", "likes",
            "always use", "default", "rather", "instead of",
        }
        query_lower = query_text.lower()
        has_pref_signal = any(sig in query_lower for sig in _PREFERENCE_SIGNALS)
        if has_pref_signal:
            for nid in list(node_scores.keys()):
                node = all_results[nid]
                etype = node.metadata.get("event_type", "")
                if etype == "user_preference":
                    node_scores[nid] *= 1.5  # Extra boost for preference matches

        # Filter expired
        for nid in list(all_results.keys()):
            if all_results[nid].is_expired():
                del all_results[nid]
                node_scores.pop(nid, None)

        # Filter superseded
        for nid in list(all_results.keys()):
            if all_results[nid].metadata.get("superseded"):
                del all_results[nid]
                node_scores.pop(nid, None)

        # Filter flagged-for-review (negative feedback threshold reached)
        for nid in list(all_results.keys()):
            if all_results[nid].metadata.get("flagged_for_review"):
                del all_results[nid]
                node_scores.pop(nid, None)

        # Filter infrastructure types
        excluded = set(exclude_types) if exclude_types else set()
        if not include_infrastructure:
            excluded |= self._INFRASTRUCTURE_TYPES
        if excluded:
            for nid in list(all_results.keys()):
                etype = all_results[nid].metadata.get("event_type", "")
                if etype in excluded:
                    del all_results[nid]
                    node_scores.pop(nid, None)

        # Session filter
        if session_id:
            for nid in list(all_results.keys()):
                node_session = all_results[nid].metadata.get("session_id", "")
                if node_session and node_session != session_id:
                    del all_results[nid]
                    node_scores.pop(nid, None)

        # Project filter
        if project_path and scope == "project":
            for nid in list(all_results.keys()):
                node_project = all_results[nid].metadata.get("project", "")
                if node_project and node_project != project_path:
                    del all_results[nid]
                    node_scores.pop(nid, None)

        # Phase 3: Contextual re-ranking
        if context_file or context_tags:
            context_set: Set[str] = set()
            if context_file:
                # Extract filename stem and path components as context signals
                from pathlib import PurePosixPath

                p = PurePosixPath(context_file)
                context_set.add(p.stem.lower())
                context_set.add(p.name.lower())
                for part in p.parts:
                    if len(part) > 2 and part not in ("/", "."):
                        context_set.add(part.lower())
            if context_tags:
                context_set.update(t.lower() for t in context_tags)

            if context_set:
                for nid in list(node_scores.keys()):
                    node = all_results[nid]
                    node_tags = set(str(t).lower() for t in (node.metadata.get("tags") or []))
                    node_project = (node.metadata.get("project") or "").lower()
                    node_content_lower = node.content.lower()

                    # Count context signal matches
                    tag_overlap = len(context_set & node_tags)
                    project_match = 1 if node_project and any(c in node_project for c in context_set) else 0
                    content_match = sum(1 for c in context_set if c in node_content_lower)

                    # Apply graduated boost: 10% per tag match, 15% for project, 5% per content hit (capped)
                    boost = 1.0 + ((tag_overlap * 0.10) + (project_match * 0.15) + (min(content_match, 3) * 0.05)) * pw_ctx
                    node_scores[nid] *= boost

        # Phase 4: Temporal constraint — in-range boost, out-of-range penalty
        if temporal_range:
            try:
                t_start, t_end = temporal_range
                for nid in list(node_scores.keys()):
                    node = all_results[nid]
                    # Only use referenced_date (explicit event time), not created_at
                    # (storage timestamp). Memories without referenced_date stay neutral.
                    ref_date = node.metadata.get("referenced_date") or ""
                    if ref_date:
                        if t_start <= ref_date <= t_end:
                            node_scores[nid] *= 1.3  # In-range boost
                        else:
                            node_scores[nid] *= 0.15  # Out-of-range penalty (soft enough to survive abstention floor)
                    # No referenced_date: leave score unchanged (neutral)
            except Exception as e:
                logger.debug("Temporal constraint failed: %s", e)

        # Entity filtering (post-scoring, same pattern as project/event_type filters)
        if entity_id:
            filtered_ids = set()
            for nid, node in all_results.items():
                node_entity = None
                if hasattr(node, "metadata") and node.metadata:
                    node_entity = node.metadata.get("entity_id")
                # Also check the entity_id column directly via a DB lookup
                if node_entity is None:
                    try:
                        row = self._conn.execute("SELECT entity_id FROM memories WHERE node_id = ?", (nid,)).fetchone()
                        if row:
                            node_entity = row[0]
                    except Exception:
                        pass
                if node_entity == entity_id:
                    filtered_ids.add(nid)
            node_scores = {k: v for k, v in node_scores.items() if k in filtered_ids}

        # Agent type filtering (post-scoring, same pattern as entity_id)
        if agent_type:
            filtered_ids = set()
            for nid, node in all_results.items():
                node_agent_type = None
                if hasattr(node, "metadata") and node.metadata:
                    node_agent_type = node.metadata.get("agent_type")
                if node_agent_type is None:
                    try:
                        row = self._conn.execute(
                            "SELECT agent_type FROM memories WHERE node_id = ?", (nid,)
                        ).fetchone()
                        if row:
                            node_agent_type = row[0]
                    except Exception:
                        pass
                if node_agent_type == agent_type:
                    filtered_ids.add(nid)
            node_scores = {k: v for k, v in node_scores.items() if k in filtered_ids}

        # Phase 4.5: Related-chain enrichment — for top results, fetch 1-hop
        # neighbors via the edge graph to surface cross-session context.
        if node_scores and limit >= 3:
            try:
                top_ids = sorted(node_scores, key=node_scores.get, reverse=True)[:3]
                for seed_id in top_ids:
                    seed_score = node_scores[seed_id]
                    neighbors = self._conn.execute(
                        """SELECT source_id, target_id, weight
                           FROM edges
                           WHERE (source_id = ? OR target_id = ?)
                           AND weight >= 0.3""",
                        (seed_id, seed_id),
                    ).fetchall()
                    for source, target, weight in neighbors:
                        nbr_id = target if source == seed_id else source
                        if nbr_id in node_scores or nbr_id in all_results:
                            continue  # already scored
                        mem_row = self._conn.execute(
                            """SELECT node_id, content, metadata, created_at,
                                      access_count, last_accessed, ttl_seconds
                               FROM memories WHERE node_id = ?""",
                            (nbr_id,),
                        ).fetchone()
                        if not mem_row:
                            continue
                        result = self._row_to_result(mem_row)
                        if result.is_expired() or result.metadata.get("superseded"):
                            continue
                        # Score neighbor at 40% of seed score, weighted by edge strength
                        nbr_score = seed_score * 0.4 * min(weight, 1.0) * pw_graph
                        if nbr_score >= self._MIN_COMPOSITE_SCORE:
                            all_results[nbr_id] = result
                            node_scores[nbr_id] = nbr_score
            except Exception as e:
                logger.debug("Related-chain enrichment failed: %s", e)

        # Phase 5 (pre): Apply plugin score modifiers
        if self._score_modifiers and node_scores:
            for nid in list(node_scores.keys()):
                meta = all_results[nid].metadata if nid in all_results else {}
                for modifier in self._score_modifiers:
                    try:
                        node_scores[nid] = modifier(nid, node_scores[nid], meta)
                    except Exception as e:
                        logger.debug("Plugin score modifier failed: %s", e)

        # Sort and dedup
        sorted_ids = sorted(node_scores.keys(), key=lambda x: node_scores[x], reverse=True)

        seen_content: Set[str] = set()
        deduped: List[MemoryResult] = []
        for nid in sorted_ids:
            node = all_results[nid]
            normalized = " ".join(node.content.lower().split())[:150]
            normalized = _TRAILING_HASH_RE.sub("", normalized)
            if normalized in seen_content:
                continue
            seen_content.add(normalized)
            deduped.append(node)
            if len(deduped) >= limit:
                break

        # Phase 5: Abstention — filter low-quality results before normalization
        if deduped:
            # Precompute query words for text-result word-overlap check
            query_words = [w for w in query_text.lower().split() if len(w) > 2]

            filtered = []
            for n in deduped:
                score = node_scores.get(n.id, 0.0)
                # Universal composite floor (catches temporal penalty, etc.)
                if score < ctx_min_composite:
                    continue
                if n.id in raw_vec_sims:
                    # Vec result: require minimum cosine similarity (dynamic #4)
                    if raw_vec_sims[n.id] >= ctx_min_vec:
                        filtered.append(n)
                    elif query_words:
                        # Fallback: vec result below threshold can survive
                        # if content + tags have strong word overlap with query
                        content_lower = n.content.lower()
                        tag_text = " ".join(str(t).lower() for t in (n.metadata.get("tags") or []))
                        searchable = content_lower + " " + tag_text
                        if self._word_overlap(query_words, searchable) >= ctx_min_text:
                            filtered.append(n)
                else:
                    # Text-only result: require minimum raw word overlap (dynamic #4)
                    if query_words:
                        content_lower = n.content.lower()
                        if self._word_overlap(query_words, content_lower) >= ctx_min_text:
                            filtered.append(n)
                    else:
                        filtered.append(n)
            deduped = filtered

        # Normalize relevance scores
        if deduped:
            max_score = max(node_scores.get(n.id, 0.0) for n in deduped)
            for node in deduped:
                raw = node_scores.get(node.id, 0.0)
                node.relevance = round(raw / max_score, 3) if max_score > 0 else 0.0

        if deduped:
            self.stats["hits"] += 1
        else:
            self.stats["misses"] += 1

        # --- Query result cache: store (tiered TTL #2) ---
        if _cache_key is not None:
            _confidence = 0.0
            if deduped:
                _confidence = sum(n.relevance for n in deduped[:3]) / min(len(deduped), 3)
            self._query_cache[_cache_key] = (now_mono, deduped, _confidence)
            while len(self._query_cache) > _QUERY_CACHE_MAX:
                self._query_cache.popitem(last=False)
        if session_id and deduped:
            self._session_cache[session_id] = deduped

        # --- A/B feedback tracking: record retrieval context for returned results ---
        for n in deduped:
            self._recent_query_context[n.id] = {
                "query_text": query_text[:200],
                "query_hint": query_hint,
                "score": round(node_scores.get(n.id, 0.0), 4),
                "vec_sim": round(raw_vec_sims.get(n.id, 0.0), 4),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            self._recent_query_context.move_to_end(n.id)
        while len(self._recent_query_context) > 50:
            self._recent_query_context.popitem(last=False)

        return deduped

    def _vec_query(self, embedding: List[float], limit: int = 10) -> List[tuple]:
        """Query the sqlite-vec virtual table. Returns [(rowid, distance), ...]."""
        if not self._vec_available:
            return []
        try:
            rows = self._conn.execute(
                "SELECT rowid, distance FROM memories_vec WHERE embedding MATCH ? AND k = ?",
                (_serialize_f32(embedding), limit),
            ).fetchall()
            return rows
        except Exception as e:
            logger.debug(f"Vec query failed: {e}")
            return []

    def _text_search(self, query_text: str, limit: int = 20) -> List[MemoryResult]:
        """Text-based search using FTS5 (fast) or LIKE fallback."""
        query_lower = query_text.lower()
        words = [w for w in query_lower.split() if len(w) > 2]
        if not words:
            return []

        # Try FTS5 first (O(log n) vs O(n) for LIKE)
        if getattr(self, "_fts_available", False):
            try:
                # FTS5 query: OR-match words, quote each to avoid syntax errors
                fts_terms = " OR ".join(f'"{w}"' for w in words)
                # Add bigram phrases for queries with 3+ words (improves precision)
                if len(words) >= 3:
                    bigrams = [f'"{words[i]} {words[i+1]}"' for i in range(len(words) - 1)]
                    fts_terms = fts_terms + " OR " + " OR ".join(bigrams)
                rows = self._conn.execute(
                    """SELECT m.node_id, m.content, m.metadata, m.created_at,
                               m.access_count, m.last_accessed, m.ttl_seconds,
                               f.rank
                        FROM memories_fts f
                        JOIN memories m ON f.rowid = m.id
                        WHERE memories_fts MATCH ?
                        ORDER BY f.rank LIMIT ?""",
                    (fts_terms, limit * 3),
                ).fetchall()

                if not rows:
                    return []

                results = []
                # BM25 rank values are negative (more negative = better match)
                ranks = [row[7] for row in rows]
                best_rank = min(ranks)  # Most negative = best
                worst_rank = max(ranks)  # Closest to 0 = worst
                rank_spread = worst_rank != best_rank

                for row in rows:
                    result = self._row_to_result(row[:7])
                    bm25_rank = row[7]
                    # Normalize BM25: best -> 1.0, worst -> 0.1
                    if rank_spread:
                        bm25_norm = 0.1 + 0.9 * (worst_rank - bm25_rank) / (worst_rank - best_rank)
                    else:
                        bm25_norm = 1.0  # Single result or all identical ranks

                    # Word-match ratio (existing logic)
                    content_lower = result.content.lower()
                    matched = sum(1 for w in words if w in content_lower)
                    word_ratio = matched / len(words)

                    # Blend: 70% BM25 (IDF-weighted) + 30% word-match
                    result.relevance = 0.7 * bm25_norm + 0.3 * word_ratio
                    results.append(result)

                results.sort(key=lambda r: r.relevance, reverse=True)
                return results[:limit]
            except Exception as e:
                logger.warning(f"FTS5 search failed: {e} — attempting auto-repair")
                try:
                    self._conn.execute("INSERT INTO memories_fts(memories_fts) VALUES('rebuild')")
                    self._commit()
                    logger.info("FTS5 index rebuilt successfully")
                    # Retry the query once after repair
                    rows = self._conn.execute(
                        """SELECT m.node_id, m.content, m.metadata, m.created_at,
                                   m.access_count, m.last_accessed, m.ttl_seconds,
                                   f.rank
                            FROM memories_fts f
                            JOIN memories m ON f.rowid = m.id
                            WHERE memories_fts MATCH ?
                            ORDER BY f.rank LIMIT ?""",
                        (fts_terms, limit * 3),
                    ).fetchall()
                    if not rows:
                        return []
                    results = []
                    ranks = [row[7] for row in rows]
                    best_rank = min(ranks)
                    worst_rank = max(ranks)
                    rank_spread = worst_rank != best_rank
                    for row in rows:
                        result = self._row_to_result(row[:7])
                        bm25_rank = row[7]
                        if rank_spread:
                            bm25_norm = 0.1 + 0.9 * (worst_rank - bm25_rank) / (worst_rank - best_rank)
                        else:
                            bm25_norm = 1.0
                        content_lower = result.content.lower()
                        matched = sum(1 for w in words if w in content_lower)
                        word_ratio = matched / len(words)
                        result.relevance = 0.7 * bm25_norm + 0.3 * word_ratio
                        results.append(result)
                    results.sort(key=lambda r: r.relevance, reverse=True)
                    return results[:limit]
                except Exception as rebuild_err:
                    logger.warning(f"FTS5 rebuild also failed: {rebuild_err} — falling back to LIKE")

        # Fallback: LIKE-based search (O(n))
        conditions = " OR ".join(["LOWER(content) LIKE ?" for _ in words])
        params = [f"%{w}%" for w in words]
        params.append(limit * 3)

        rows = self._conn.execute(
            f"""SELECT node_id, content, metadata, created_at,
                       access_count, last_accessed, ttl_seconds
                FROM memories WHERE ({conditions})
                ORDER BY created_at DESC LIMIT ?""",
            params,
        ).fetchall()

        results = []
        for row in rows:
            result = self._row_to_result(row)
            content_lower = result.content.lower()
            matched = sum(1 for w in words if w in content_lower)
            result.relevance = matched / len(words)
            results.append(result)

        results.sort(key=lambda r: r.relevance, reverse=True)
        return results[:limit]

    # ------------------------------------------------------------------
    # Index-style lookups (replacing TypeIndex, SessionIndex)
    # ------------------------------------------------------------------

    def get_by_type(self, event_type: str, limit: int = 100) -> List[MemoryResult]:
        """Get memories by event type, sorted by recency."""
        rows = self._conn.execute(
            """SELECT node_id, content, metadata, created_at,
                      access_count, last_accessed, ttl_seconds
               FROM memories WHERE event_type = ?
               ORDER BY created_at DESC LIMIT ?""",
            (event_type, limit),
        ).fetchall()
        return [self._row_to_result(row) for row in rows]

    def get_by_session(self, session_id: str, limit: int = 100) -> List[MemoryResult]:
        """Get memories by session ID, sorted by recency."""
        rows = self._conn.execute(
            """SELECT node_id, content, metadata, created_at,
                      access_count, last_accessed, ttl_seconds
               FROM memories WHERE session_id = ?
               ORDER BY created_at DESC LIMIT ?""",
            (session_id, limit),
        ).fetchall()
        return [self._row_to_result(row) for row in rows]

    def query_by_type(
        self,
        query: str,
        event_type: str,
        limit: int = 10,
        min_similarity: float = 0.3,
        project_path: str = "",
        scope: str = "project",
    ) -> List[MemoryResult]:
        """Search within a specific event type using embeddings or text."""
        # Try vector search first
        if self._vec_available:
            try:
                from omega.graphs import generate_embedding

                query_emb = generate_embedding(query)
                if query_emb:
                    vec_results = self._vec_query(query_emb, limit=limit * 5)
                    results = []
                    for rowid, distance in vec_results:
                        similarity = 1.0 - distance
                        if similarity < min_similarity:
                            continue
                        row = self._conn.execute(
                            """SELECT node_id, content, metadata, created_at,
                                      access_count, last_accessed, ttl_seconds
                               FROM memories WHERE id = ? AND event_type = ?""",
                            (rowid, event_type),
                        ).fetchone()
                        if row:
                            result = self._row_to_result(row)
                            # Project filter
                            if project_path and scope == "project":
                                node_project = result.metadata.get("project", "")
                                if node_project and node_project != project_path:
                                    continue
                            result.relevance = similarity
                            results.append(result)
                            if len(results) >= limit:
                                break
                    return results
            except Exception as e:
                logger.debug("Type-filtered vec search failed, falling back to text: %s", e)

        # Fallback: text search within type
        query_lower = query.lower()
        words = [w for w in query_lower.split() if len(w) > 2]
        if not words:
            return self.get_by_type(event_type, limit)

        conditions = " AND ".join(["LOWER(content) LIKE ?" for _ in words[:3]])
        params = [event_type] + [f"%{w}%" for w in words[:3]]
        params.append(limit)

        rows = self._conn.execute(
            f"""SELECT node_id, content, metadata, created_at,
                       access_count, last_accessed, ttl_seconds
                FROM memories WHERE event_type = ? AND ({conditions})
                ORDER BY created_at DESC LIMIT ?""",
            params,
        ).fetchall()

        results = []
        for row in rows:
            result = self._row_to_result(row)
            content_lower = result.content.lower()
            matched = sum(1 for w in words if w in content_lower)
            result.relevance = matched / len(words)
            results.append(result)
        return results

    def get_type_stats(self) -> Dict[str, int]:
        """Get counts for all event types."""
        rows = self._conn.execute(
            "SELECT event_type, COUNT(*) FROM memories WHERE event_type IS NOT NULL GROUP BY event_type"
        ).fetchall()
        return {row[0]: row[1] for row in rows}

    def get_session_stats(self) -> Dict[str, int]:
        """Get counts for all sessions."""
        rows = self._conn.execute(
            "SELECT session_id, COUNT(*) FROM memories WHERE session_id IS NOT NULL GROUP BY session_id"
        ).fetchall()
        return {row[0]: row[1] for row in rows}

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    def clear_session(self, session_id: str) -> int:
        """Clear all memories for a session. Returns count removed."""
        with self._lock:
            # Capture IDs BEFORE deleting memories
            rows = self._conn.execute("SELECT id, node_id FROM memories WHERE session_id = ?", (session_id,)).fetchall()

            if not rows:
                return 0

            rowids = [r[0] for r in rows]
            node_ids = [r[1] for r in rows]

            # Delete memories
            self._conn.execute("DELETE FROM memories WHERE session_id = ?", (session_id,))

            # Delete vec embeddings
            if self._vec_available:
                for rid in rowids:
                    try:
                        self._conn.execute("DELETE FROM memories_vec WHERE rowid = ?", (rid,))
                    except Exception as e:
                        logger.debug("Failed to delete vec embedding rowid=%s: %s", rid, e)

            # Clean up edges referencing deleted nodes
            if node_ids:
                placeholders = ",".join("?" * len(node_ids))
                self._conn.execute(
                    f"DELETE FROM edges WHERE source_id IN ({placeholders}) OR target_id IN ({placeholders})",
                    node_ids + node_ids,
                )

            self._commit()
            return len(rows)

    # ------------------------------------------------------------------
    # Memory management
    # ------------------------------------------------------------------

    def cleanup_expired(self) -> int:
        """Remove expired memories. Returns count removed."""
        with self._lock:
            now = datetime.now(timezone.utc).isoformat()
            # Find expired: created_at + ttl_seconds < now
            rows = self._conn.execute(
                """SELECT id, node_id FROM memories
                   WHERE ttl_seconds IS NOT NULL
                   AND datetime(created_at, '+' || ttl_seconds || ' seconds') < ?""",
                (now,),
            ).fetchall()

            if not rows:
                return 0

            for rowid, node_id in rows:
                self._conn.execute("DELETE FROM memories WHERE id = ?", (rowid,))
                if self._vec_available:
                    try:
                        self._conn.execute("DELETE FROM memories_vec WHERE rowid = ?", (rowid,))
                    except Exception as e:
                        logger.debug("Failed to delete vec embedding rowid=%s: %s", rowid, e)
                self._conn.execute("DELETE FROM edges WHERE source_id = ? OR target_id = ?", (node_id, node_id))

            self._commit()
            return len(rows)

    def evict_lru(self, count: int = 1) -> int:
        """Evict least recently used memories."""
        with self._lock:
            rows = self._conn.execute(
                """SELECT id, node_id FROM memories
                   ORDER BY COALESCE(last_accessed, created_at) ASC
                   LIMIT ?""",
                (count,),
            ).fetchall()

            evicted = 0
            for rowid, node_id in rows:
                self._conn.execute("DELETE FROM memories WHERE id = ?", (rowid,))
                if self._vec_available:
                    try:
                        self._conn.execute("DELETE FROM memories_vec WHERE rowid = ?", (rowid,))
                    except Exception as e:
                        logger.debug("Failed to delete vec embedding rowid=%s: %s", rowid, e)
                self._conn.execute("DELETE FROM edges WHERE source_id = ? OR target_id = ?", (node_id, node_id))
                evicted += 1

            if evicted:
                self._commit()
            return evicted

    def consolidate(
        self,
        prune_days: int = 30,
        max_summaries: int = 50,
    ) -> Dict[str, Any]:
        """Consolidate memories: prune stale low-value entries, cap session summaries.

        Prunes:
        1. Memories with 0 access older than prune_days (excluding protected types)
        2. Oldest session summaries beyond max_summaries cap
        3. Orphaned edges pointing to deleted nodes
        4. Orphaned vec embeddings without matching memory rows

        Returns dict with counts of what was removed.
        """
        protected_types = frozenset(
            {
                "user_preference",
                "lesson_learned",
                "error_pattern",
                "decision",
            }
        )
        stats = {"pruned_stale": 0, "pruned_summaries": 0, "pruned_edges": 0, "pruned_vec_orphans": 0}

        cutoff = (datetime.now(timezone.utc) - timedelta(days=prune_days)).isoformat()

        with self._lock:
            # Phase 1: Prune stale zero-access memories (not protected types)
            placeholders = ",".join("?" * len(protected_types))
            rows = self._conn.execute(
                f"""SELECT id, node_id FROM memories
                    WHERE access_count = 0
                    AND created_at < ?
                    AND (event_type IS NULL OR event_type NOT IN ({placeholders}))""",
                (cutoff, *protected_types),
            ).fetchall()

            for rowid, node_id in rows:
                self._conn.execute("DELETE FROM memories WHERE id = ?", (rowid,))
                if self._vec_available:
                    try:
                        self._conn.execute("DELETE FROM memories_vec WHERE rowid = ?", (rowid,))
                    except Exception as e:
                        logger.debug("Failed to delete vec embedding rowid=%s: %s", rowid, e)
                self._conn.execute("DELETE FROM edges WHERE source_id = ? OR target_id = ?", (node_id, node_id))
                stats["pruned_stale"] += 1

            # Phase 2: Cap session summaries — keep newest max_summaries, prune rest
            summary_rows = self._conn.execute(
                """SELECT id, node_id FROM memories
                   WHERE event_type = 'session_summary'
                   ORDER BY created_at DESC"""
            ).fetchall()

            if len(summary_rows) > max_summaries:
                to_prune = summary_rows[max_summaries:]
                for rowid, node_id in to_prune:
                    self._conn.execute("DELETE FROM memories WHERE id = ?", (rowid,))
                    if self._vec_available:
                        try:
                            self._conn.execute("DELETE FROM memories_vec WHERE rowid = ?", (rowid,))
                        except Exception as e:
                            logger.debug("Vec cleanup during summary prune failed for rowid %s: %s", rowid, e)
                    self._conn.execute("DELETE FROM edges WHERE source_id = ? OR target_id = ?", (node_id, node_id))
                    stats["pruned_summaries"] += 1

            # Phase 3: Prune orphaned edges
            orphaned = self._conn.execute(
                """SELECT e.id FROM edges e
                   LEFT JOIN memories m1 ON e.source_id = m1.node_id
                   LEFT JOIN memories m2 ON e.target_id = m2.node_id
                   WHERE m1.node_id IS NULL OR m2.node_id IS NULL"""
            ).fetchall()
            if orphaned:
                self._conn.execute(
                    f"DELETE FROM edges WHERE id IN ({','.join('?' * len(orphaned))})", [r[0] for r in orphaned]
                )
                stats["pruned_edges"] = len(orphaned)

            # Phase 4: Prune orphaned vec embeddings
            if self._vec_available:
                try:
                    orphaned_vec = self._conn.execute(
                        """SELECT vec.rowid FROM memories_vec vec
                           LEFT JOIN memories m ON vec.rowid = m.id
                           WHERE m.id IS NULL"""
                    ).fetchall()
                    if orphaned_vec:
                        for row in orphaned_vec:
                            try:
                                self._conn.execute("DELETE FROM memories_vec WHERE rowid = ?", (row[0],))
                            except Exception as e:
                                logger.debug("Vec orphan delete failed for rowid %s: %s", row[0], e)
                        stats["pruned_vec_orphans"] = len(orphaned_vec)
                        logger.info("Pruned %d orphaned vec embeddings", len(orphaned_vec))
                except Exception as e:
                    logger.debug("Vec orphan check failed: %s", e)

            self._commit()

        stats["node_count_after"] = self.node_count()
        return stats

    # ------------------------------------------------------------------
    # Batch operations
    # ------------------------------------------------------------------

    def batch_store(self, items: List[Dict[str, Any]]) -> List[str]:
        """Store multiple memories efficiently."""
        if not items:
            return []

        # Batch-generate embeddings for items without them
        items_needing = [(i, item) for i, item in enumerate(items) if item.get("embedding") is None]
        if items_needing:
            try:
                from omega.graphs import generate_embeddings_batch, get_active_backend

                texts = [item["content"] for _, item in items_needing]
                embeddings = generate_embeddings_batch(texts)
                backend = get_active_backend()
                if backend is not None:
                    # Real ML embeddings — store in vec table
                    for (idx, item), emb in zip(items_needing, embeddings):
                        item["embedding"] = emb
                else:
                    # Hash fallback — do NOT store in vec table (incompatible with ML embeddings)
                    logger.warning(
                        f"batch_store: skipping {len(texts)} embeddings (hash fallback — "
                        f"would corrupt vector search). Memories will be findable via text search only."
                    )
            except Exception as e:
                logger.warning(f"batch_store: embedding generation failed: {e}")

        ids = []
        for item in items:
            node_id = self.store(
                content=item["content"],
                session_id=item.get("session_id"),
                metadata=item.get("metadata"),
                embedding=item.get("embedding"),
                dependencies=item.get("dependencies"),
                ttl_seconds=item.get("ttl_seconds"),
            )
            ids.append(node_id)

        return ids

    def reembed_all(self, batch_size: int = 32) -> Dict[str, int]:
        """Regenerate all embeddings using the current ML model.

        Use this to fix corrupted (hash-fallback) embeddings or after
        switching embedding models. Only runs if an ML backend is available.

        Returns dict with counts of updated, skipped, and failed nodes.
        """
        from omega.graphs import generate_embeddings_batch, get_active_backend

        backend = get_active_backend()
        if backend is None:
            # Force a load attempt
            from omega.graphs import _get_embedding_model

            _get_embedding_model()
            backend = get_active_backend()
        if backend is None:
            raise RuntimeError("Cannot reembed: no ML embedding backend available")

        rows = self._conn.execute("SELECT id, content FROM memories ORDER BY id").fetchall()

        updated = 0
        failed = 0
        for i in range(0, len(rows), batch_size):
            batch_rows = rows[i : i + batch_size]
            texts = [r[1] for r in batch_rows]
            ids = [r[0] for r in batch_rows]

            try:
                embeddings = generate_embeddings_batch(texts)
                with self._lock:
                    for mem_id, emb in zip(ids, embeddings):
                        try:
                            self._conn.execute("DELETE FROM memories_vec WHERE rowid = ?", (mem_id,))
                            self._conn.execute(
                                "INSERT INTO memories_vec (rowid, embedding) VALUES (?, ?)",
                                (mem_id, _serialize_f32(emb)),
                            )
                            updated += 1
                        except Exception as e:
                            logger.warning(f"reembed failed for id={mem_id}: {e}")
                            failed += 1
                    self._commit()
            except Exception as e:
                logger.warning(f"reembed batch failed: {e}")
                failed += len(batch_rows)

        logger.info(f"reembed_all: updated={updated}, failed={failed}")
        return {"updated": updated, "failed": failed, "total": len(rows)}

    # ------------------------------------------------------------------
    # Health / status
    # ------------------------------------------------------------------

    def check_memory_health(
        self,
        warn_mb: float = 350,
        critical_mb: float = 800,
        max_nodes: int = 10000,
    ) -> Dict[str, Any]:
        """Check memory health. Returns health dict."""
        count = self.node_count()
        db_size_mb = self.db_path.stat().st_size / (1024 * 1024) if self.db_path.exists() else 0

        # Estimate process RSS (ru_maxrss is bytes on macOS, KB on Linux)
        try:
            import resource
            import sys as _sys

            rss_raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            if _sys.platform == "darwin":
                rss_mb = rss_raw / (1024 * 1024)  # bytes → MB
            else:
                rss_mb = rss_raw / 1024  # KB → MB
        except (ImportError, OSError):
            rss_mb = 0

        status = "healthy"
        warnings = []
        recommendations = []

        if rss_mb > critical_mb:
            status = "critical"
            warnings.append(
                f"RSS memory at {rss_mb:.0f} MB (critical threshold: {critical_mb} MB). "
                "Note: ONNX embedding model loads ~300 MB into memory on first query "
                "and auto-unloads after 10 min idle. This is expected peak usage."
            )
        elif rss_mb > warn_mb:
            status = "warning"
            warnings.append(
                f"RSS memory at {rss_mb:.0f} MB (warn threshold: {warn_mb} MB). "
                "Note: ONNX embedding model loads ~300 MB on first query; "
                "auto-unloads after 10 min idle."
            )

        if count > max_nodes:
            warnings.append(f"Node count {count} exceeds max {max_nodes}")
            recommendations.append("Run omega consolidate to deduplicate and prune")

        return {
            "status": status,
            "memory_mb": rss_mb,
            "db_size_mb": round(db_size_mb, 2),
            "node_count": count,
            "warnings": warnings,
            "recommendations": recommendations,
            "usage": {
                "stores": self.stats.get("stores", 0),
                "queries": self.stats.get("queries", 0),
                "vec_enabled": self._vec_available,
            },
        }

    # ------------------------------------------------------------------
    # Feedback
    # ------------------------------------------------------------------

    def record_feedback(self, node_id: str, rating: str, reason: Optional[str] = None) -> Dict[str, Any]:
        """Record feedback on a memory node."""
        with self._lock:
            row = self._conn.execute("SELECT metadata FROM memories WHERE node_id = ?", (node_id,)).fetchone()
            if not row:
                return {"error": f"Memory node {node_id} not found"}

            meta = json.loads(row[0]) if row[0] else {}

            if "feedback_signals" not in meta:
                meta["feedback_signals"] = []
            if "feedback_score" not in meta:
                meta["feedback_score"] = 0

            score_delta = {"helpful": 1, "unhelpful": -1, "outdated": -2}.get(rating, 0)
            meta["feedback_score"] += score_delta
            signal = {
                "rating": rating,
                "reason": reason,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            # Attach retrieval context if available (A/B tracking)
            retrieval_ctx = self._recent_query_context.get(node_id)
            if retrieval_ctx:
                signal["retrieval_context"] = retrieval_ctx
            meta["feedback_signals"].append(signal)

            if meta["feedback_score"] <= -3:
                meta["flagged_for_review"] = True

            self._conn.execute("UPDATE memories SET metadata = ? WHERE node_id = ?", (json.dumps(meta), node_id))
            self._commit()

            return {
                "node_id": node_id,
                "rating": rating,
                "new_score": meta["feedback_score"],
                "total_signals": len(meta["feedback_signals"]),
                "flagged": meta.get("flagged_for_review", False),
                "cache_invalidated": 0,
            }

    def add_edge(self, source_id: str, target_id: str, edge_type: str = "related", weight: float = 1.0) -> bool:
        """Insert an edge between two memories (thread-safe, idempotent)."""
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            try:
                self._conn.execute(
                    """INSERT OR IGNORE INTO edges
                       (source_id, target_id, edge_type, weight, created_at)
                       VALUES (?, ?, ?, ?, ?)""",
                    (source_id, target_id, edge_type, round(weight, 3), now),
                )
                self._commit()
                return True
            except Exception as e:
                logger.debug(f"add_edge failed: {e}")
                return False

    def get_related_chain(
        self,
        start_id: str,
        max_hops: int = 2,
        min_weight: float = 0.0,
        edge_types: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Traverse relationship edges from a starting memory up to max_hops.

        Returns a list of dicts with: node_id, content, hop, weight, edge_type, path.
        Nodes are deduplicated (each appears at its shortest hop distance).
        """
        if max_hops < 1 or max_hops > 5:
            max_hops = min(max(max_hops, 1), 5)

        visited: Dict[str, Dict[str, Any]] = {}
        frontier = {start_id}

        for hop in range(1, max_hops + 1):
            if not frontier:
                break
            next_frontier: Set[str] = set()
            for node_id in frontier:
                # Query edges in both directions (undirected graph)
                rows = self._conn.execute(
                    """SELECT source_id, target_id, edge_type, weight
                       FROM edges
                       WHERE (source_id = ? OR target_id = ?)
                       AND weight >= ?""",
                    (node_id, node_id, min_weight),
                ).fetchall()

                for source, target, etype, weight in rows:
                    neighbor = target if source == node_id else source
                    if neighbor == start_id or neighbor in visited:
                        continue
                    if edge_types and etype not in edge_types:
                        continue

                    # Fetch the memory content
                    mem_row = self._conn.execute(
                        """SELECT node_id, content, metadata, created_at,
                                  access_count, last_accessed, ttl_seconds
                           FROM memories WHERE node_id = ?""",
                        (neighbor,),
                    ).fetchone()
                    if not mem_row:
                        continue

                    result = self._row_to_result(mem_row)
                    visited[neighbor] = {
                        "node_id": neighbor,
                        "content": result.content,
                        "metadata": result.metadata,
                        "created_at": result.created_at.isoformat() if result.created_at else "",
                        "hop": hop,
                        "weight": weight,
                        "edge_type": etype,
                    }
                    next_frontier.add(neighbor)

            frontier = next_frontier

        # Sort by hop (nearest first), then by weight (strongest first)
        results = sorted(visited.values(), key=lambda x: (x["hop"], -x["weight"]))
        return results

    # ------------------------------------------------------------------
    # Export / Import
    # ------------------------------------------------------------------

    def export_to_file(self, filepath: Path) -> Dict[str, Any]:
        """Export all memories to a JSON file."""
        rows = self._conn.execute(
            """SELECT node_id, content, metadata, created_at,
                      access_count, last_accessed, ttl_seconds
               FROM memories ORDER BY created_at"""
        ).fetchall()

        nodes = []
        sessions = set()
        for row in rows:
            result = self._row_to_result(row)
            nodes.append(
                {
                    "id": result.id,
                    "content": result.content,
                    "metadata": result.metadata,
                    "created_at": result.created_at.isoformat(),
                    "access_count": result.access_count,
                    "last_accessed": result.last_accessed.isoformat() if result.last_accessed else None,
                    "ttl_seconds": result.ttl_seconds,
                }
            )
            sid = result.metadata.get("session_id")
            if sid:
                sessions.add(sid)

        export_data = {
            "version": "omega-sqlite-v1",
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "node_count": len(nodes),
            "session_count": len(sessions),
            "nodes": nodes,
        }

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        # Write with restricted permissions (0o600) — export contains plaintext memories
        export_bytes = json.dumps(export_data, indent=2).encode("utf-8")
        fd = os.open(str(filepath), os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o600)
        try:
            os.write(fd, export_bytes)
        finally:
            os.close(fd)

        return {
            "filepath": str(filepath),
            "node_count": len(nodes),
            "session_count": len(sessions),
            "file_size_kb": filepath.stat().st_size / 1024,
            "exported_at": export_data["exported_at"],
        }

    def import_from_file(self, filepath: Path, clear_existing: bool = True) -> Dict[str, Any]:
        """Import memories from a JSON file."""
        data = json.loads(Path(filepath).read_text())
        nodes = data.get("nodes", [])

        if clear_existing:
            self._conn.execute("DELETE FROM memories")
            if self._vec_available:
                try:
                    self._conn.execute("DELETE FROM memories_vec")
                except Exception as e:
                    logger.debug("Vec table clear during import failed: %s", e)
            self._conn.execute("DELETE FROM edges")
            self._commit()

        imported = 0
        for node_data in nodes:
            try:
                self.store(
                    content=node_data["content"],
                    session_id=node_data.get("metadata", {}).get("session_id"),
                    metadata=node_data.get("metadata"),
                    ttl_seconds=node_data.get("ttl_seconds"),
                    skip_inference=True,
                )
                imported += 1
            except Exception as e:
                logger.debug(f"Import failed for node: {e}")

        return {
            "filepath": str(filepath),
            "node_count": imported,
            "session_count": data.get("session_count", 0),
        }

    def get_session_context(self, session_id: str, limit: int = 50, include_recent: bool = True) -> List[MemoryResult]:
        """Get context for a session."""
        results = {}
        for node in self.get_by_session(session_id, limit=limit):
            results[node.id] = node

        if include_recent and len(results) < limit:
            remaining = limit - len(results)
            recent = self.get_recent(limit=remaining)
            for node in recent:
                if node.id not in results:
                    results[node.id] = node

        return list(results.values())[:limit]

    def get_recent(self, limit: int = 10) -> List[MemoryResult]:
        """Get most recent memories."""
        rows = self._conn.execute(
            """SELECT node_id, content, metadata, created_at,
                      access_count, last_accessed, ttl_seconds
               FROM memories ORDER BY created_at DESC LIMIT ?""",
            (limit,),
        ).fetchall()
        return [self._row_to_result(row) for row in rows]

    def get_embedding(self, node_id: str) -> Optional[List[float]]:
        """Retrieve the stored embedding for a node."""
        if not self._vec_available:
            return None
        row = self._conn.execute("SELECT id FROM memories WHERE node_id = ?", (node_id,)).fetchone()
        if not row:
            return None
        rowid = row[0]
        vec_row = self._conn.execute("SELECT embedding FROM memories_vec WHERE rowid = ?", (rowid,)).fetchone()
        if not vec_row:
            return None
        return _deserialize_f32(vec_row[0])

    def find_similar(self, embedding: List[float], limit: int = 10) -> List[MemoryResult]:
        """Find semantically similar memories by embedding."""
        if not self._vec_available or not embedding:
            return []

        vec_results = self._vec_query(embedding, limit=limit)
        results = []
        for rowid, distance in vec_results:
            row = self._conn.execute(
                """SELECT node_id, content, metadata, created_at,
                          access_count, last_accessed, ttl_seconds
                   FROM memories WHERE id = ?""",
                (rowid,),
            ).fetchone()
            if row:
                result = self._row_to_result(row)
                result.relevance = 1.0 - distance
                results.append(result)
        return results

    def get_timeline(self, days: int = 7, limit_per_day: int = 10) -> Dict[str, List[MemoryResult]]:
        """Get memories grouped by date for the last N days."""
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        rows = self._conn.execute(
            """SELECT node_id, content, metadata, created_at,
                      access_count, last_accessed, ttl_seconds
               FROM memories
               WHERE created_at >= ?
               ORDER BY created_at DESC""",
            (cutoff,),
        ).fetchall()
        timeline: Dict[str, List[MemoryResult]] = {}
        for row in rows:
            result = self._row_to_result(row)
            day = result.created_at.strftime("%Y-%m-%d")
            if day not in timeline:
                timeline[day] = []
            if len(timeline[day]) < limit_per_day:
                timeline[day].append(result)
        return timeline

    def phrase_search(
        self,
        phrase: str,
        case_sensitive: bool = False,
        event_type: Optional[str] = None,
        limit: int = 10,
        project_path: str = "",
        scope: str = "project",
    ) -> List[MemoryResult]:
        """Exact substring search across memories."""
        conditions = []
        params = []

        if case_sensitive:
            conditions.append("content LIKE ?")
            params.append(f"%{phrase}%")
        else:
            conditions.append("LOWER(content) LIKE ?")
            params.append(f"%{phrase.lower()}%")

        if event_type:
            conditions.append("event_type = ?")
            params.append(event_type)

        if project_path and scope == "project":
            conditions.append("(project IS NULL OR project = '' OR project = ?)")
            params.append(project_path)

        params.append(limit)

        rows = self._conn.execute(
            f"""SELECT node_id, content, metadata, created_at,
                       access_count, last_accessed, ttl_seconds
                FROM memories WHERE {" AND ".join(conditions)}
                ORDER BY created_at DESC LIMIT ?""",
            params,
        ).fetchall()
        return [self._row_to_result(row) for row in rows]

    # ------------------------------------------------------------------
    # Stats persistence
    # ------------------------------------------------------------------

    def _load_stats(self) -> None:
        """Load stats from a sidecar file if it exists."""
        stats_path = self.db_path.parent / "stats.json"
        if stats_path.exists():
            try:
                loaded = json.loads(stats_path.read_text())
                self.stats.update(loaded)
            except (json.JSONDecodeError, OSError) as e:
                logger.debug("Stats load failed: %s", e)

    def _save_stats(self) -> None:
        """Persist stats to a sidecar file."""
        stats_path = self.db_path.parent / "stats.json"
        try:
            self.stats["total_nodes"] = self.node_count()
            stats_path.write_text(json.dumps(self.stats))
        except (OSError, TypeError) as e:
            logger.debug("Stats save failed: %s", e)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    # Pre-compiled regex for keyword detection
    _CAMELCASE_RE = re.compile(r'\b[A-Z][a-z]+[A-Z]\w*\b')
    _FILEPATH_RE = re.compile(r'/[a-zA-Z][\w/.]*')

    @staticmethod
    def _is_keyword_sufficient(query_text: str) -> bool:
        """Detect if a query is keyword-driven enough to skip vector embedding.

        Conservative heuristics — only skips when the query clearly contains
        code identifiers, file paths, or quoted phrases where semantic search
        adds latency without improving results.
        """
        # Contains backticks (code spans)
        if '`' in query_text:
            return True
        # Contains file paths (/foo/bar or ./baz)
        if SQLiteStore._FILEPATH_RE.search(query_text):
            return True
        # Quoted phrase search
        stripped = query_text.strip()
        if stripped.startswith('"') and stripped.endswith('"') and len(stripped) > 2:
            return True
        # CamelCase identifiers (e.g., SQLiteStore, MemoryResult)
        if SQLiteStore._CAMELCASE_RE.search(query_text):
            return True
        return False

    @staticmethod
    def _word_overlap(query_words: list, searchable: str) -> float:
        """Compute word overlap ratio with lightweight stemming and canonicalization.

        Checks exact substring match first, then falls back to
        suffix-stripped stems to handle morphological variants
        (e.g., deploy/deployed/deployment all share stem 'deploy').
        Applies NFKC canonicalization (#6) for better matching.
        """
        if not query_words:
            return 0.0
        searchable = _canonicalize(searchable)
        _SUFFIXES = (
            "ation",
            "tion",
            "ment",
            "ing",
            "ness",
            "ity",
            "ous",
            "ive",
            "able",
            "ed",
            "er",
            "es",
            "ly",
            "al",
            "s",
        )
        matched = 0
        for w in query_words:
            cw = _canonicalize(w)
            if cw in searchable:
                matched += 1
            else:
                # Lightweight stemming: strip one common suffix
                stem = cw
                for suffix in _SUFFIXES:
                    if cw.endswith(suffix) and len(cw) - len(suffix) >= 3:
                        stem = cw[: -len(suffix)]
                        break
                if stem != cw and stem in searchable:
                    matched += 1
        return matched / len(query_words)

    @staticmethod
    def _compute_fb_factor(fb_score: int) -> float:
        """Compute feedback factor for query scoring.

        Amplified formula: positive feedback gives meaningful boost,
        negative feedback aggressively demotes.
        """
        if fb_score >= 0:
            return 1.0 + min(fb_score, 10) * 0.15  # +5 → 1.75x, +10 → 2.5x (capped)
        else:
            return max(0.2, 1.0 + fb_score * 0.2)  # -2 → 0.6x, -4 → 0.2x (floor)

    @staticmethod
    def _parse_dt(value: Optional[str]) -> Optional[datetime]:
        """Parse an ISO datetime string to an aware UTC datetime.

        Handles naive strings (no tz), Z-suffix, and +00:00 suffix.
        Returns None when *value* is falsy.
        """
        if not value:
            return None
        # Python 3.11+ fromisoformat supports 'Z' natively, but we keep this
        # workaround for existing DB records written with the Z-suffix format.
        if value.endswith("Z"):
            value = value[:-1] + "+00:00"
        dt = datetime.fromisoformat(value)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt

    def _row_to_result(self, row: tuple) -> MemoryResult:
        """Convert a database row to a MemoryResult."""
        node_id, content, metadata_json, created_at, access_count, last_accessed, ttl_seconds = row

        meta = json.loads(metadata_json) if metadata_json else {}

        created = self._parse_dt(created_at) or datetime.now(timezone.utc)
        last_acc = self._parse_dt(last_accessed)

        return MemoryResult(
            id=node_id,
            content=content,
            metadata=meta,
            created_at=created,
            access_count=access_count or 0,
            last_accessed=last_acc,
            ttl_seconds=ttl_seconds,
        )


    # ------------------------------------------------------------------
    # Engram-inspired improvements
    # ------------------------------------------------------------------

    def _fast_path_lookup(self, query_text: str, limit: int = 10) -> List[MemoryResult]:
        """Hash-based fast-path lookup (#1): O(1) trigram fingerprint match."""
        query_fp = _trigram_fingerprint(query_text)
        if not query_fp or len(query_fp) < 5:
            return []
        if not self._is_keyword_sufficient(query_text):
            return []

        matches: List[Tuple[float, MemoryResult]] = []
        for nid, mem in self._hot_memories.items():
            mem_fp = _trigram_fingerprint(mem.content)
            sim = _trigram_jaccard(query_fp, mem_fp)
            if sim >= _FAST_PATH_MIN_OVERLAP:
                matches.append((sim, mem))

        if len(matches) < limit:
            try:
                query_lower = query_text.lower()
                words = [w for w in query_lower.split() if len(w) > 2]
                if words:
                    conditions = " AND ".join(["LOWER(content) LIKE ?" for _ in words[:3]])
                    params = [f"%{w}%" for w in words[:3]]
                    params.append(limit * 3)
                    rows = self._conn.execute(
                        f"""SELECT node_id, content, metadata, created_at,
                                   access_count, last_accessed, ttl_seconds
                            FROM memories WHERE ({conditions})
                            ORDER BY access_count DESC LIMIT ?""",
                        params,
                    ).fetchall()
                    seen_ids = {m[1].id for m in matches}
                    for row in rows:
                        result = self._row_to_result(row)
                        if result.id in seen_ids:
                            continue
                        if result.is_expired() or result.metadata.get("superseded"):
                            continue
                        mem_fp = _trigram_fingerprint(result.content)
                        sim = _trigram_jaccard(query_fp, mem_fp)
                        if sim >= _FAST_PATH_MIN_OVERLAP:
                            matches.append((sim, result))
                            seen_ids.add(result.id)
            except Exception as e:
                logger.debug("Fast-path SQL lookup failed: %s", e)

        if not matches:
            return []
        matches.sort(key=lambda x: x[0], reverse=True)
        results = []
        for sim, mem in matches[:limit]:
            mem.relevance = round(sim, 3)
            results.append(mem)
        return results

    def _check_hot_tier(self, query_text: str, limit: int = 10) -> List[MemoryResult]:
        """Check hot memory tier (#2) for quick matches."""
        if not self._hot_memories:
            return []
        query_words = [w for w in query_text.lower().split() if len(w) > 2]
        if not query_words:
            return []
        matches: List[Tuple[float, MemoryResult]] = []
        for nid, mem in self._hot_memories.items():
            if mem.is_expired() or mem.metadata.get("superseded"):
                continue
            overlap = self._word_overlap(query_words, mem.content.lower())
            if overlap >= 0.4:
                matches.append((overlap, mem))
        if not matches:
            return []
        matches.sort(key=lambda x: x[0], reverse=True)
        results = []
        for overlap, mem in matches[:limit]:
            result = MemoryResult(
                id=mem.id, content=mem.content, metadata=mem.metadata,
                created_at=mem.created_at, access_count=mem.access_count,
                last_accessed=mem.last_accessed, ttl_seconds=mem.ttl_seconds,
                relevance=round(overlap, 3),
            )
            results.append(result)
        return results

    def _refresh_hot_cache(self) -> None:
        """Refresh the hot memory cache (#2) with top memories by access_count."""
        try:
            rows = self._conn.execute(
                """SELECT node_id, content, metadata, created_at,
                          access_count, last_accessed, ttl_seconds
                   FROM memories WHERE access_count > 0
                   ORDER BY access_count DESC LIMIT ?""",
                (_HOT_CACHE_SIZE,),
            ).fetchall()
            new_hot: Dict[str, MemoryResult] = {}
            for row in rows:
                result = self._row_to_result(row)
                if not result.is_expired():
                    new_hot[result.id] = result
            self._hot_memories = new_hot
            self._hot_cache_ts = _time.monotonic()
        except Exception as e:
            logger.debug("Hot cache refresh failed: %s", e)

    def _classify_query_intent(self, query_text: str) -> Optional[QueryIntent]:
        """Classify query intent for adaptive retrieval budget (#3)."""
        if self._is_keyword_sufficient(query_text):
            return QueryIntent.NAVIGATIONAL
        query_lower = query_text.lower()
        _FACTUAL_SIGNALS = (
            "what was", "what is", "what are", "which", "when did", "when was",
            "who", "where", "did we", "did i", "was there", "is there",
            "decision about", "preference for", "error with", "bug in",
            "remind me", "remember",
        )
        if any(query_lower.startswith(sig) or sig in query_lower for sig in _FACTUAL_SIGNALS):
            return QueryIntent.FACTUAL
        _CONCEPTUAL_SIGNALS = (
            "how does", "how do", "how to", "why does", "why do", "why is",
            "explain", "understand", "overview", "architecture", "design",
            "pattern", "approach", "strategy", "concept",
        )
        if any(query_lower.startswith(sig) or sig in query_lower for sig in _CONCEPTUAL_SIGNALS):
            return QueryIntent.CONCEPTUAL
        return None

    def clear_session_cache(self, session_id: str) -> None:
        """Clear session affinity cache (#2) when session ends."""
        self._session_cache.pop(session_id, None)

    def prefetch_for_project(self, project_path: str, file_stems: Optional[List[str]] = None) -> int:
        """Prefetch memories for a project's key files (#5)."""
        if not file_stems:
            try:
                rows = self._conn.execute(
                    """SELECT content FROM memories
                       WHERE project = ? AND access_count > 0
                       ORDER BY access_count DESC LIMIT 100""",
                    (project_path,),
                ).fetchall()
                import re as _re
                file_pattern = _re.compile(r'\b[\w/.-]+\.\w{1,5}\b')
                file_counts: Dict[str, int] = {}
                for row in rows:
                    for match in file_pattern.findall(row[0]):
                        stem = Path(match).stem
                        if len(stem) > 2:
                            file_counts[stem] = file_counts.get(stem, 0) + 1
                file_stems = sorted(file_counts, key=file_counts.get, reverse=True)[:10]
            except Exception as e:
                logger.debug("Prefetch file extraction failed: %s", e)
                return 0
        if not file_stems:
            return 0
        total_prefetched = 0
        for stem in file_stems:
            try:
                rows = self._conn.execute(
                    """SELECT node_id, content, metadata, created_at,
                              access_count, last_accessed, ttl_seconds
                       FROM memories WHERE LOWER(content) LIKE ?
                       AND (project = ? OR project IS NULL)
                       ORDER BY access_count DESC LIMIT 10""",
                    (f"%{stem.lower()}%", project_path),
                ).fetchall()
                results = []
                for row in rows:
                    result = self._row_to_result(row)
                    if not result.is_expired() and not result.metadata.get("superseded"):
                        results.append(result)
                if results:
                    self._prefetch_cache[stem] = results
                    total_prefetched += len(results)
            except Exception as e:
                logger.debug("Prefetch for stem '%s' failed: %s", stem, e)
        return total_prefetched

    def close(self) -> None:
        """Close the database connection."""
        self._save_stats()
        try:
            self._conn.close()
        except Exception as e:
            logger.debug("Database close failed: %s", e)

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass  # Silence errors during GC — no logger guarantee
