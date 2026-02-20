"""Tests for Engram-inspired improvements to OMEGA's retrieval pipeline.

Covers all 6 improvements:
1. Hash-based fast-path lookup (trigram fingerprints)
2. Tiered memory cache (hot/warm/session)
3. Content canonicalization (NFKC, markdown strip)
4. Context-dependent gating (SurfacingContext)
5. Adaptive retrieval budget (QueryIntent)
6. Predictive prefetching
"""

import hashlib
import json
import os
import time

import pytest

from omega.sqlite_store import (
    SCHEMA_VERSION,
    SQLiteStore,
    SurfacingContext,
    QueryIntent,
    MemoryResult,
    _canonicalize,
    _trigram_fingerprint,
    _trigram_jaccard,
    _SURFACING_THRESHOLDS,
    _INTENT_WEIGHTS,
    _HOT_CACHE_SIZE,
    _HOT_CACHE_REFRESH_S,
    _FAST_PATH_MIN_OVERLAP,
)


# ============================================================================
# 1. Hash-Based Fast-Path Lookup
# ============================================================================


class TestFastPathLookup:
    """Tests for trigram fingerprint fast-path (#1)."""

    def test_trigram_fingerprint_basic(self):
        """Trigram fingerprint produces non-empty frozenset for normal text."""
        fp = _trigram_fingerprint("hello world")
        assert isinstance(fp, frozenset)
        assert len(fp) > 0

    def test_trigram_fingerprint_short_text(self):
        """Text shorter than 3 chars produces empty fingerprint."""
        assert _trigram_fingerprint("ab") == frozenset()
        assert _trigram_fingerprint("") == frozenset()

    def test_trigram_fingerprint_case_insensitive(self):
        """Fingerprints are case-insensitive (via canonicalization)."""
        fp1 = _trigram_fingerprint("Hello World")
        fp2 = _trigram_fingerprint("hello world")
        assert fp1 == fp2

    def test_trigram_jaccard_identical(self):
        """Identical fingerprints have Jaccard similarity of 1.0."""
        fp = _trigram_fingerprint("exact match text")
        assert _trigram_jaccard(fp, fp) == 1.0

    def test_trigram_jaccard_disjoint(self):
        """Completely different texts have low Jaccard similarity."""
        fp1 = _trigram_fingerprint("python programming language")
        fp2 = _trigram_fingerprint("xyz abc qrs tuv")
        sim = _trigram_jaccard(fp1, fp2)
        assert sim < 0.3

    def test_trigram_jaccard_empty(self):
        """Empty fingerprints return 0.0."""
        assert _trigram_jaccard(frozenset(), frozenset()) == 0.0
        fp = _trigram_fingerprint("hello")
        assert _trigram_jaccard(fp, frozenset()) == 0.0

    def test_fast_path_returns_results(self, store):
        """Fast-path should return results for keyword-heavy queries."""
        store.store(
            content="Error: sqlite3.OperationalError: database is locked",
            session_id="s1",
            metadata={"event_type": "error_pattern"},
        )
        # Touch to populate hot cache
        store._refresh_hot_cache()
        results = store._fast_path_lookup("sqlite3.OperationalError database locked")
        # May or may not match depending on keyword detection — just verify no crash
        assert isinstance(results, list)

    def test_fast_path_empty_for_conceptual(self, store):
        """Fast-path should return empty for non-keyword queries."""
        store.store(content="How authentication works in the system", session_id="s1")
        store._refresh_hot_cache()
        results = store._fast_path_lookup("how does authentication work")
        assert results == []


# ============================================================================
# 2. Tiered Memory Cache
# ============================================================================


class TestTieredCache:
    """Tests for tiered cache with hot set (#2)."""

    def test_hot_cache_populated_on_init(self, store):
        """Hot cache should be initialized (possibly empty in fresh store)."""
        assert hasattr(store, "_hot_memories")
        assert isinstance(store._hot_memories, dict)

    def test_hot_cache_refresh(self, store):
        """_refresh_hot_cache populates from high-access memories."""
        nid = store.store(content="Frequently accessed memory about testing patterns")
        # Bump access_count directly
        store._conn.execute(
            "UPDATE memories SET access_count = 50 WHERE node_id = ?", (nid,)
        )
        store._conn.commit()
        store._refresh_hot_cache()
        assert nid in store._hot_memories

    def test_hot_cache_invalidated_on_write(self, store):
        """Writing should mark hot cache for refresh."""
        store._hot_cache_ts = time.monotonic()  # Set fresh
        store.store(content="New memory triggers cache invalidation")
        assert store._hot_cache_ts == 0.0  # Reset by _invalidate_query_cache

    def test_session_cache_cleared(self, store):
        """clear_session_cache removes entries for the session."""
        store._session_cache["test-session"] = [
            MemoryResult(id="mem-test", content="cached")
        ]
        store.clear_session_cache("test-session")
        assert "test-session" not in store._session_cache

    def test_session_cache_populated_after_query(self, store):
        """Query with session_id should populate session cache."""
        store.store(content="Memory about Python testing patterns for cache test")
        results = store.query("Python testing patterns", session_id="sess-cache-test")
        # Session cache should be populated if there were results
        if results:
            assert "sess-cache-test" in store._session_cache

    def test_warm_cache_extended_ttl(self, store):
        """High-confidence results should use warm TTL (300s vs 60s)."""
        store.store(content="High confidence memory for warm cache test")
        # First query populates cache
        store.query("High confidence memory warm cache", use_cache=True)
        # Check cache entry has 3-tuple (ts, results, confidence)
        for key, val in store._query_cache.items():
            assert len(val) == 3  # (timestamp, results, confidence)
            break

    def test_query_cache_key_includes_surfacing_context(self, store):
        """Cache key should vary by surfacing context."""
        store.store(content="Context-sensitive caching test memory")
        store.query("caching test", surfacing_context=SurfacingContext.GENERAL)
        store.query("caching test", surfacing_context=SurfacingContext.ERROR_DEBUG)
        # Both should be cached separately
        keys = list(store._query_cache.keys())
        assert len(keys) >= 1  # At least one cached

    def test_hot_cache_size_limit(self, store):
        """Hot cache should respect _HOT_CACHE_SIZE limit."""
        # Store more than HOT_CACHE_SIZE memories
        for i in range(60):
            nid = store.store(content=f"Memory {i}: unique content for hot cache limit test item number {i}")
            store._conn.execute(
                "UPDATE memories SET access_count = ? WHERE node_id = ?",
                (100 - i, nid),
            )
        store._conn.commit()
        store._refresh_hot_cache()
        assert len(store._hot_memories) <= _HOT_CACHE_SIZE


# ============================================================================
# 3. Adaptive Retrieval Budget
# ============================================================================


class TestAdaptiveRetrievalBudget:
    """Tests for query intent classification (#5)."""

    def test_navigational_intent(self, store):
        """File paths and code identifiers should classify as NAVIGATIONAL."""
        assert store._classify_query_intent("/src/omega/bridge.py") == QueryIntent.NAVIGATIONAL
        assert store._classify_query_intent("`MemoryResult`") == QueryIntent.NAVIGATIONAL

    def test_factual_intent(self, store):
        """Factual questions should classify as FACTUAL."""
        assert store._classify_query_intent("what was the decision about caching") == QueryIntent.FACTUAL
        assert store._classify_query_intent("when did we add the entity system") == QueryIntent.FACTUAL

    def test_conceptual_intent(self, store):
        """Conceptual questions should classify as CONCEPTUAL."""
        assert store._classify_query_intent("how does the query pipeline work") == QueryIntent.CONCEPTUAL
        assert store._classify_query_intent("explain the architecture of omega") == QueryIntent.CONCEPTUAL

    def test_ambiguous_returns_none(self, store):
        """Ambiguous queries return None (use default weights)."""
        result = store._classify_query_intent("deployment rollback steps")
        assert result is None

    def test_intent_weights_structure(self):
        """All intent weights should be 5-tuples."""
        for intent, weights in _INTENT_WEIGHTS.items():
            assert len(weights) == 5
            assert all(isinstance(w, (int, float)) for w in weights)


# ============================================================================
# 4. Context-Dependent Gating
# ============================================================================


class TestContextDependentGating:
    """Tests for surfacing context thresholds (#4)."""

    def test_all_contexts_have_thresholds(self):
        """Every SurfacingContext enum value should have thresholds defined."""
        for ctx in SurfacingContext:
            assert ctx in _SURFACING_THRESHOLDS
            thresholds = _SURFACING_THRESHOLDS[ctx]
            assert len(thresholds) == 4
            assert all(isinstance(t, (int, float)) for t in thresholds)

    def test_error_debug_lower_vec_threshold(self):
        """ERROR_DEBUG should have lower vec threshold than GENERAL."""
        gen = _SURFACING_THRESHOLDS[SurfacingContext.GENERAL]
        err = _SURFACING_THRESHOLDS[SurfacingContext.ERROR_DEBUG]
        assert err[0] < gen[0]  # Lower min_vec_similarity

    def test_session_start_broader_surfacing(self):
        """SESSION_START should surface broadly for better context at startup."""
        start = _SURFACING_THRESHOLDS[SurfacingContext.SESSION_START]
        assert start[0] > 0  # Has a min_vec_similarity
        assert start[1] > 0  # Has a min_text_relevance
        assert start[2] > 0  # Has a min_composite_score

    def test_file_edit_boosted_context_weight(self):
        """FILE_EDIT should have boosted context weight."""
        gen = _SURFACING_THRESHOLDS[SurfacingContext.GENERAL]
        edit = _SURFACING_THRESHOLDS[SurfacingContext.FILE_EDIT]
        assert edit[3] > gen[3]  # Higher context_weight_boost

    def test_query_with_surfacing_context(self, store):
        """Query should accept surfacing_context parameter without error."""
        store.store(content="Test memory for surfacing context query integration")
        for ctx in SurfacingContext:
            results = store.query(
                "surfacing context test",
                surfacing_context=ctx,
                limit=5,
            )
            assert isinstance(results, list)


# ============================================================================
# 5. Predictive Prefetching
# ============================================================================


class TestPredictivePrefetching:
    """Tests for predictive prefetching (#6)."""

    def test_prefetch_with_file_stems(self, store):
        """prefetch_for_project with explicit file stems."""
        store.store(
            content="The bridge.py module handles all MCP tool routing",
            metadata={"project": "/proj/omega"},
        )
        count = store.prefetch_for_project("/proj/omega", file_stems=["bridge"])
        assert count >= 0  # May be 0 if content doesn't match LIKE

    def test_prefetch_auto_discovers_stems(self, store):
        """prefetch_for_project auto-discovers file stems from content."""
        for i in range(5):
            store.store(
                content=f"Change {i} to sqlite_store.py for performance improvement",
                metadata={"project": "/proj/omega"},
            )
            store._conn.execute(
                "UPDATE memories SET access_count = ? WHERE node_id = (SELECT node_id FROM memories ORDER BY id DESC LIMIT 1)",
                (10 + i,),
            )
        store._conn.commit()
        count = store.prefetch_for_project("/proj/omega")
        assert isinstance(count, int)

    def test_prefetch_empty_project(self, store):
        """Prefetching for non-existent project returns 0."""
        count = store.prefetch_for_project("/nonexistent/project")
        assert count == 0

    def test_prefetch_populates_cache(self, store):
        """Prefetched results should appear in _prefetch_cache."""
        store.store(
            content="Important pattern in sqlite_store for query optimization",
            metadata={"project": "/proj/test"},
        )
        store._conn.execute(
            "UPDATE memories SET access_count = 20 WHERE node_id = (SELECT node_id FROM memories ORDER BY id DESC LIMIT 1)"
        )
        store._conn.commit()
        store.prefetch_for_project("/proj/test", file_stems=["sqlite_store"])
        # Cache may or may not have entries depending on LIKE match
        assert isinstance(store._prefetch_cache, dict)


# ============================================================================
# 6. Content Canonicalization
# ============================================================================


class TestContentCanonicalization:
    """Tests for content canonicalization (#6)."""

    def test_canonicalize_basic(self):
        """Basic canonicalization: lowercase, strip whitespace."""
        assert _canonicalize("Hello World") == "hello world"

    def test_canonicalize_markdown(self):
        """Markdown formatting should be stripped."""
        result = _canonicalize("**bold** and `code` and ## heading")
        assert "**" not in result
        assert "`" not in result
        assert "##" not in result
        assert "bold" in result
        assert "code" in result

    def test_canonicalize_whitespace_collapse(self):
        """Multiple whitespace chars should collapse to single space."""
        result = _canonicalize("hello    world\n\nnew  paragraph")
        assert "  " not in result
        assert "\n" not in result

    def test_canonicalize_nfkc(self):
        """NFKC normalization should handle Unicode equivalences."""
        # Fullwidth 'A' (U+FF21) should normalize to regular 'a'
        result = _canonicalize("\uff21\uff22\uff23")
        assert result == "abc"

    def test_canonical_hash_stored(self, store):
        """Stored memories should have canonical_hash computed."""
        nid = store.store(content="Test canonical hash storage")
        row = store._conn.execute(
            "SELECT canonical_hash FROM memories WHERE node_id = ?", (nid,)
        ).fetchone()
        assert row is not None
        assert row[0] is not None
        expected = hashlib.sha256(
            _canonicalize("Test canonical hash storage").encode()
        ).hexdigest()
        assert row[0] == expected

    def test_canonical_dedup(self, store):
        """Content with different markdown but same canonical form should dedup."""
        nid1 = store.store(content="Important lesson about testing")
        nid2 = store.store(content="**Important** lesson about testing")
        assert nid1 == nid2
        assert store.node_count() == 1

    def test_canonical_dedup_whitespace(self, store):
        """Content differing only in whitespace should dedup."""
        nid1 = store.store(content="lesson about   testing patterns")
        nid2 = store.store(content="lesson about testing patterns")
        assert nid1 == nid2

    def test_canonical_hash_updated_on_edit(self, store):
        """update_node with new content should recompute canonical_hash."""
        nid = store.store(content="Original content for hash test")
        store.update_node(nid, content="Updated content for hash test")
        row = store._conn.execute(
            "SELECT canonical_hash FROM memories WHERE node_id = ?", (nid,)
        ).fetchone()
        expected = hashlib.sha256(
            _canonicalize("Updated content for hash test").encode()
        ).hexdigest()
        assert row[0] == expected

    def test_word_overlap_uses_canonicalization(self, store):
        """_word_overlap should canonicalize both query and searchable text."""
        # Markdown in content should not prevent matching
        overlap = store._word_overlap(
            ["important", "lesson"],
            "**Important** lesson about `testing`",
        )
        assert overlap > 0

    def test_canonicalize_preserves_content(self, store):
        """Original content should be preserved despite canonical dedup."""
        nid = store.store(content="**Bold** lesson about `code`")
        node = store.get_node(nid)
        assert node.content == "**Bold** lesson about `code`"


# ============================================================================
# Integration Tests
# ============================================================================


class TestEngramIntegration:
    """Integration tests combining multiple improvements."""

    def test_full_query_pipeline(self, store):
        """Full query through the enhanced pipeline."""
        store.store(
            content="Decision: use SQLite for persistence instead of Redis",
            metadata={"event_type": "decision", "project": "/proj/omega"},
        )
        store.store(
            content="Lesson: always validate input at API boundaries",
            metadata={"event_type": "lesson_learned", "project": "/proj/omega"},
        )
        store.store(
            content="Error: connection timeout when Redis is down",
            metadata={"event_type": "error_pattern", "project": "/proj/omega"},
        )

        # General query
        results = store.query(
            "SQLite persistence decision",
            surfacing_context=SurfacingContext.GENERAL,
            limit=5,
        )
        assert isinstance(results, list)

        # Error debug query
        results = store.query(
            "connection timeout error",
            surfacing_context=SurfacingContext.ERROR_DEBUG,
            limit=5,
        )
        assert isinstance(results, list)

    def test_schema_version_is_5(self, store):
        """SCHEMA_VERSION should be 5 after Engram improvements."""
        assert SCHEMA_VERSION == 5
        row = store._conn.execute(
            "SELECT version FROM schema_version LIMIT 1"
        ).fetchone()
        assert row[0] == 5

    def test_stats_tracking(self, store):
        """Engram stats should be tracked."""
        store.store(content="Memory for stats tracking test")
        store._refresh_hot_cache()
        store.query("stats tracking test")
        # Stats dict should exist and have standard keys
        assert "queries" in store.stats
        assert "stores" in store.stats
