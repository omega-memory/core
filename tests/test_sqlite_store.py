"""Tests for OMEGA SQLiteStore — the core storage backend."""
import json
import os
import pytest


class TestFlaggedMemoryFiltering:
    """Flagged memories should be excluded from query results."""

    def test_flagged_memory_excluded_from_query(self, store):
        """Memories with flagged_for_review=True should not appear in query results."""
        nid = store.store(
            content="This is a bad memory that got flagged for having wrong info about the API",
            metadata={"event_type": "lesson_learned"},
        )
        # Flag it via direct metadata update (simulating feedback_score <= -3)
        node = store.get_node(nid)
        meta = node.metadata
        meta["feedback_score"] = -4
        meta["flagged_for_review"] = True
        store._conn.execute(
            "UPDATE memories SET metadata = ? WHERE node_id = ?",
            (json.dumps(meta), nid),
        )
        store._conn.commit()

        # Query should NOT return the flagged memory
        results = store.query("bad memory wrong info API", limit=10)
        result_ids = [r.id for r in results]
        assert nid not in result_ids

    def test_unflagged_memory_appears_in_query(self, store):
        """Normal memories should still appear in query results."""
        nid = store.store(
            content="This is a perfectly good lesson about testing patterns in Python",
            metadata={"event_type": "lesson_learned"},
        )
        results = store.query("testing patterns Python", limit=10)
        result_ids = [r.id for r in results]
        assert nid in result_ids


class TestStoreBasics:
    """Core CRUD operations."""

    def test_store_and_retrieve(self, store):
        nid = store.store(content="Hello world", session_id="s1")
        assert nid.startswith("mem-")
        assert store.node_count() == 1

        node = store.get_node(nid)
        assert node is not None
        assert node.content == "Hello world"
        assert node.metadata.get("session_id") == "s1"

    def test_store_dedup_exact(self, store):
        nid1 = store.store(content="Duplicate content")
        nid2 = store.store(content="Duplicate content")
        assert nid1 == nid2
        assert store.node_count() == 1

    def test_delete_node(self, store):
        nid = store.store(content="To be deleted")
        assert store.node_count() == 1
        result = store.delete_node(nid)
        assert result is True
        assert store.node_count() == 0
        assert store.get_node(nid) is None

    def test_delete_nonexistent(self, store):
        result = store.delete_node("nonexistent-id")
        assert result is False

    def test_update_node(self, store):
        nid = store.store(content="Original content")
        store.update_node(nid, content="Updated content")
        node = store.get_node(nid)
        assert node.content == "Updated content"

    def test_update_metadata(self, store):
        nid = store.store(content="Test", metadata={"key": "value1"})
        store.update_node(nid, metadata={"key": "value2", "new_key": "new_value"})
        node = store.get_node(nid)
        assert node.metadata["key"] == "value2"
        assert node.metadata["new_key"] == "new_value"


class TestQuery:
    """Search and retrieval."""

    def test_text_search(self, store):
        store.store(content="Python is a programming language")
        store.store(content="JavaScript runs in browsers")
        store.store(content="Rust is a systems language")

        results = store.query("programming language", limit=5)
        assert len(results) >= 1
        # Python entry should match
        contents = [r.content for r in results]
        assert any("Python" in c for c in contents)

    def test_query_empty_store(self, store):
        results = store.query("anything", limit=5)
        assert results == []

    def test_get_by_type(self, store):
        store.store(content="A decision", metadata={"event_type": "decision"})
        store.store(content="A lesson", metadata={"event_type": "lesson_learned"})
        store.store(content="Another decision", metadata={"event_type": "decision"})

        decisions = store.get_by_type("decision", limit=10)
        assert len(decisions) == 2
        assert all(r.metadata.get("event_type") == "decision" for r in decisions)

    def test_get_by_session(self, store):
        store.store(content="Session A memory", session_id="session-a")
        store.store(content="Session B memory", session_id="session-b")
        store.store(content="Session A again", session_id="session-a")

        results = store.get_by_session("session-a", limit=10)
        assert len(results) == 2

    def test_get_recent(self, store):
        for i in range(5):
            store.store(content=f"Memory {i}")
        results = store.get_recent(limit=3)
        assert len(results) == 3

    def test_phrase_search(self, store):
        store.store(content="The quick brown fox jumps over the lazy dog")
        store.store(content="A fox in the wild")

        results = store.phrase_search("brown fox", limit=5)
        assert len(results) >= 1
        assert "brown fox" in results[0].content


class TestTTL:
    """Time-to-live and expiration."""

    def test_expired_node(self, store):
        nid = store.store(content="Ephemeral", ttl_seconds=0)
        node = store.get_node(nid)
        # With TTL=0, it should be immediately expired
        assert node is not None  # Still retrievable
        # But cleanup should remove it
        removed = store.cleanup_expired()
        assert removed >= 1

    def test_permanent_node(self, store):
        nid = store.store(content="Permanent", ttl_seconds=None)
        node = store.get_node(nid)
        assert node.ttl_seconds is None
        assert not node.is_expired()


class TestBatchOps:
    """Batch operations."""

    def test_batch_store(self, store):
        # Use very distinct content to avoid embedding-based dedup with hash fallback
        items = [
            {"content": f"Unique topic number {i}: {'abcdefghij'[i]} is for {'apple banana cherry date elderberry fig grape honeydew iris jackfruit'.split()[i]}", "session_id": "batch"}
            for i in range(10)
        ]
        ids = store.batch_store(items)
        assert len(ids) == 10
        # Some may dedup via embedding similarity with hash fallback, so just check we got IDs back
        assert all(isinstance(nid, str) for nid in ids)

    def test_find_similar(self, store):
        store.store(content="Python programming language")
        store.store(content="JavaScript programming language")
        store.store(content="Cooking recipes for dinner")

        # find_similar takes an embedding vector, not text — generate one
        from omega.graphs import generate_embedding
        emb = generate_embedding("Python code")
        results = store.find_similar(emb, limit=5)
        # Should return results
        assert len(results) >= 1


class TestPersistence:
    """Database persistence."""

    def test_data_survives_reopen(self, tmp_omega_dir):
        from omega.sqlite_store import SQLiteStore
        db_path = tmp_omega_dir / "persist.db"

        # Write
        s1 = SQLiteStore(db_path=db_path)
        nid = s1.store(content="Persistent memory")
        s1.close()

        # Read back
        s2 = SQLiteStore(db_path=db_path)
        assert s2.node_count() == 1
        node = s2.get_node(nid)
        assert node.content == "Persistent memory"
        s2.close()


class TestEdgeCases:
    """Edge cases and error handling."""

    def test_empty_content(self, store):
        # Empty content should be rejected with ValueError
        with pytest.raises(ValueError):
            store.store(content="")

    def test_large_content(self, store):
        big = "x" * 100_000
        nid = store.store(content=big)
        node = store.get_node(nid)
        assert len(node.content) == 100_000

    def test_unicode_content(self, store):
        nid = store.store(content="Hello 世界 🌍 مرحبا")
        node = store.get_node(nid)
        assert "世界" in node.content

    def test_special_chars_in_metadata(self, store):
        meta = {"key": "value with 'quotes' and \"double quotes\""}
        nid = store.store(content="Test", metadata=meta)
        node = store.get_node(nid)
        assert "quotes" in node.metadata["key"]

    def test_concurrent_access(self, tmp_omega_dir):
        """Two store instances can read/write the same DB (WAL mode)."""
        from omega.sqlite_store import SQLiteStore
        db_path = tmp_omega_dir / "concurrent.db"

        s1 = SQLiteStore(db_path=db_path)
        s2 = SQLiteStore(db_path=db_path)

        nid = s1.store(content="Written by s1")
        node = s2.get_node(nid)
        assert node is not None
        assert node.content == "Written by s1"

        s1.close()
        s2.close()


class TestConsolidateVecOrphans:
    """Tests for Phase 4 of consolidate() — orphaned vec embedding cleanup."""

    def test_consolidate_prunes_vec_orphans(self, store):
        """consolidate() removes vec entries without matching memory rows."""
        nid = store.store(content="Test orphan cleanup")
        row = store._conn.execute(
            "SELECT id FROM memories WHERE node_id = ?", (nid,)
        ).fetchone()
        rowid = row[0]
        # Delete memory but leave vec entry (simulate silent vec delete failure)
        store._conn.execute("DELETE FROM memories WHERE id = ?", (rowid,))
        store._conn.execute(
            "DELETE FROM edges WHERE source_id = ? OR target_id = ?", (nid, nid)
        )
        store._conn.commit()
        # Vec entry is now orphaned
        if store._vec_available:
            vec_count = store._conn.execute(
                "SELECT COUNT(*) FROM memories_vec WHERE rowid = ?", (rowid,)
            ).fetchone()[0]
            assert vec_count == 1  # Orphan exists
            result = store.consolidate()
            assert result.get("pruned_vec_orphans", 0) >= 1
            vec_count = store._conn.execute(
                "SELECT COUNT(*) FROM memories_vec WHERE rowid = ?", (rowid,)
            ).fetchone()[0]
            assert vec_count == 0  # Orphan cleaned

    def test_consolidate_no_vec_orphans(self, store):
        """consolidate() reports 0 when no orphans exist."""
        store.store(content="Clean memory")
        result = store.consolidate()
        assert result.get("pruned_vec_orphans", 0) == 0


class TestEvictLru:
    """Tests for SQLiteStore.evict_lru."""

    def test_evict_lru_removes_oldest(self, store):
        """evict_lru removes the least-recently-used memory."""
        nid1 = store.store(content="First memory stored")
        nid2 = store.store(content="Second memory stored")
        assert store.node_count() == 2

        evicted = store.evict_lru(count=1)
        assert evicted == 1
        assert store.node_count() == 1
        # First stored (oldest) should be evicted
        assert store.get_node(nid1) is None
        assert store.get_node(nid2) is not None

    def test_evict_lru_zero_count(self, store):
        """evict_lru with count=0 evicts nothing."""
        store.store(content="Keep this memory")
        evicted = store.evict_lru(count=0)
        assert evicted == 0
        assert store.node_count() == 1

    def test_evict_lru_empty_store(self, store):
        """evict_lru on empty store returns 0."""
        evicted = store.evict_lru(count=5)
        assert evicted == 0


class TestStatsPersistence:
    """Tests for _save_stats / _load_stats round-trip."""

    def test_stats_round_trip(self, tmp_omega_dir):
        from omega.sqlite_store import SQLiteStore
        db_path = tmp_omega_dir / "stats_test.db"

        s1 = SQLiteStore(db_path=db_path)
        s1.store(content="Bump the store counter")
        assert s1.stats["stores"] >= 1
        stores_before = s1.stats["stores"]
        s1.close()  # _save_stats called on close

        # Re-open — should load persisted stats
        s2 = SQLiteStore(db_path=db_path)
        assert s2.stats["stores"] == stores_before
        s2.close()

    def test_stats_load_missing_file(self, tmp_omega_dir):
        """_load_stats with no sidecar file doesn't crash."""
        from omega.sqlite_store import SQLiteStore
        db_path = tmp_omega_dir / "no_stats.db"
        s = SQLiteStore(db_path=db_path)
        assert s.stats["stores"] == 0
        s.close()


class TestCheckMemoryHealth:
    """Tests for SQLiteStore.check_memory_health."""

    def test_healthy_store(self, store):
        store.store(content="A healthy memory")
        # Use high thresholds — test process has ONNX model + Docling loaded (~2GB+)
        health = store.check_memory_health(warn_mb=4000, critical_mb=8000)
        assert health["status"] == "healthy"
        assert health["node_count"] == 1
        assert health["db_size_mb"] >= 0
        assert health["usage"]["stores"] >= 1
        assert isinstance(health["warnings"], list)

    def test_health_returns_vec_status(self, store):
        health = store.check_memory_health()
        assert "vec_enabled" in health["usage"]


class TestRecordFeedback:
    """Tests for SQLiteStore.record_feedback."""

    def test_helpful_feedback_increments_score(self, store):
        nid = store.store(content="A memory that deserves feedback for being useful")
        result = store.record_feedback(nid, "helpful", reason="Very relevant")
        assert result["new_score"] == 1
        assert result["total_signals"] == 1
        assert result["flagged"] is False

    def test_unhelpful_feedback_decrements_score(self, store):
        nid = store.store(content="A memory that will receive negative feedback")
        store.record_feedback(nid, "unhelpful")
        result = store.record_feedback(nid, "unhelpful")
        assert result["new_score"] == -2
        assert result["total_signals"] == 2

    def test_feedback_flags_at_threshold(self, store):
        nid = store.store(content="A memory that will be flagged after repeated negative feedback")
        store.record_feedback(nid, "outdated")  # -2
        result = store.record_feedback(nid, "unhelpful")  # -3
        assert result["new_score"] == -3
        assert result["flagged"] is True

    def test_feedback_nonexistent_node(self, store):
        result = store.record_feedback("nonexistent-id", "helpful")
        assert "error" in result


class TestCircuitBreakerCooldown:
    """Tests for time-based circuit breaker recovery in graphs.py."""

    def test_circuit_breaker_cooldown_recovery(self):
        """After cooldown expires, circuit breaker allows fresh attempts."""
        from unittest.mock import patch
        from omega.graphs import (
            _get_embedding_model, reset_embedding_state,
            _time_module,
        )

        reset_embedding_state()
        os.environ["OMEGA_SKIP_EMBEDDINGS"] = "1"
        # Use controlled fake time to avoid issues on CI where monotonic()
        # may be small (fresh runner), making backdated values negative.
        fake_time = [10000.0]
        try:
            with patch.object(_time_module, "monotonic", lambda: fake_time[0]):
                # Exhaust 3 attempts
                for _ in range(3):
                    _get_embedding_model()
                assert _get_embedding_model() is None  # Tripped

                # Advance time past the 5-minute cooldown
                fake_time[0] = 10301.0

                # Should allow fresh attempt (returns None because SKIP is set,
                # but counter resets)
                _get_embedding_model()
                assert _get_embedding_model._attempt_count == 1  # Fresh cycle started
        finally:
            os.environ.pop("OMEGA_SKIP_EMBEDDINGS", None)
            reset_embedding_state()

    def test_circuit_breaker_stays_tripped_before_cooldown(self):
        """Before cooldown expires, circuit breaker remains tripped."""
        from omega.graphs import _get_embedding_model, reset_embedding_state

        reset_embedding_state()
        os.environ["OMEGA_SKIP_EMBEDDINGS"] = "1"
        try:
            # Exhaust 3 attempts
            for _ in range(3):
                _get_embedding_model()
            assert _get_embedding_model() is None  # Tripped

            # _FIRST_FAILURE_TIME is recent — should stay tripped
            assert _get_embedding_model() is None
            assert _get_embedding_model._attempt_count == 3
        finally:
            os.environ.pop("OMEGA_SKIP_EMBEDDINGS", None)
            reset_embedding_state()
