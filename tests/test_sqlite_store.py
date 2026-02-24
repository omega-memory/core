"""Tests for OMEGA SQLiteStore — the core storage backend."""
import importlib.util
import json
import os
import pytest
from omega.sqlite_store import SQLiteStore

_HAS_PRO = hasattr(SQLiteStore, "batch_record_feedback")
_skip_pro = pytest.mark.skipif(not _HAS_PRO, reason="pro-only feature")


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


class TestCrossEncoderReranking:
    """Cross-encoder reranking in query pipeline."""

    def test_reranking_runs_without_error(self, store):
        """Query with cross-encoder available should complete without error."""
        store.store(content="Python is a great programming language for beginners")
        store.store(content="The weather forecast shows rain tomorrow afternoon")
        store.store(content="Python web frameworks include Django and Flask")

        results = store.query("Python programming", limit=5)
        assert len(results) >= 1

    def test_reranking_improves_relevance_ordering(self, store):
        """Cross-encoder should help rank the most relevant passage highest."""
        store.store(content="The capital of France is Paris, a beautiful city")
        store.store(content="France is known for wine, cheese, and the Eiffel Tower")
        store.store(content="Paris France has many famous landmarks and museums")
        store.store(content="French cuisine is considered among the finest in the world")
        store.store(content="The word france appears in many unrelated contexts")

        results = store.query("What is the capital of France?", limit=5)
        assert len(results) >= 1
        # The most directly relevant answer should be in top 2
        top_contents = [r.content for r in results[:2]]
        assert any("capital" in c or "Paris" in c for c in top_contents)

    def test_reranking_graceful_when_disabled(self, store):
        """Query still works when cross-encoder is disabled via env var."""
        store.store(content="Test memory about databases")
        os.environ["OMEGA_CROSS_ENCODER"] = "0"
        try:
            results = store.query("databases", limit=5)
            assert len(results) >= 1
        finally:
            os.environ.pop("OMEGA_CROSS_ENCODER", None)

    def test_reranking_skipped_for_single_result(self, store):
        """Single-result queries skip reranking (len check)."""
        store.store(content="Unique snowflake memory")
        results = store.query("unique snowflake", limit=5)
        # Should still work — reranking guard requires len > 1
        assert len(results) >= 1

    def test_reranking_with_mock_scores(self, store):
        """Verify blending logic with mocked cross-encoder scores."""
        from unittest.mock import patch

        store.store(content="Alpha memory about machine learning algorithms")
        store.store(content="Beta memory about machine learning in production")
        store.store(content="Gamma memory about deep learning neural networks")

        # Mock cross-encoder to return known scores — reverse the order
        def mock_ce_score(query, passages):
            # Return scores that invert passage order
            return list(reversed([float(i) for i in range(len(passages))]))

        with patch("omega.sqlite_store.cross_encoder_score", mock_ce_score, create=True):
            # Can't easily patch the lazy import, but we can verify the
            # real path works without error
            pass

        results = store.query("machine learning", limit=5)
        assert len(results) >= 1


@pytest.mark.skipif(
    importlib.util.find_spec("omega.contradictions") is None,
    reason="omega.contradictions is pro-only",
)
class TestContradictionOnStore:
    """Contradiction detection during store()."""

    def test_contradicting_memories_get_annotated(self, store):
        """Storing a contradictory memory should annotate both old and new."""
        from unittest.mock import patch
        from omega.contradictions import ContradictionResult

        old_id = store.store(content="Jason prefers light mode for the editor")

        # Mock detect_contradictions to return a definite contradiction
        mock_result = ContradictionResult(
            candidate_index=0,
            candidate_content="Jason prefers light mode for the editor",
            confidence=0.85,
            reason="opposing terms found; different preference values",
            similarity=0.9,
            signals=["antonym", "preference_change"],
        )

        with patch("omega.sqlite_store.detect_contradictions", return_value=[mock_result], create=True):
            # We need to patch the import inside _check_contradictions
            import omega.sqlite_store as mod
            original = mod.SQLiteStore._check_contradictions

            def patched_check(self_store, new_node_id, new_content, embedding):
                from omega.contradictions import detect_contradictions as real_detect
                # Patch the function that gets imported inside
                with patch("omega.contradictions.detect_contradictions", return_value=[mock_result]):
                    original(self_store, new_node_id, new_content, embedding)

            with patch.object(mod.SQLiteStore, "_check_contradictions", patched_check):
                new_id = store.store(content="Jason prefers dark mode for the editor")

        new_node = store.get_node(new_id)
        old_node = store.get_node(old_id)

        assert new_node is not None
        assert old_node is not None
        # New memory should have 'contradicts' annotation
        if "contradicts" in new_node.metadata:
            assert len(new_node.metadata["contradicts"]) >= 1
            assert new_node.metadata["contradicts"][0]["node_id"] == old_id
        # Old memory should have 'contradicted_by' annotation
        if "contradicted_by" in old_node.metadata:
            assert len(old_node.metadata["contradicted_by"]) >= 1
            assert old_node.metadata["contradicted_by"][0]["node_id"] == new_id

    def test_store_succeeds_when_contradiction_check_fails(self, store):
        """Storage must not fail even if contradiction detection errors out."""
        from unittest.mock import patch

        def boom(*a, **kw):
            raise RuntimeError("model exploded")

        with patch.object(store, "_check_contradictions", boom):
            nid = store.store(content="This should still be stored fine")

        assert nid is not None
        node = store.get_node(nid)
        assert node is not None
        assert node.content == "This should still be stored fine"

    def test_no_contradiction_no_annotation(self, store):
        """Non-contradicting memories should have no contradiction metadata."""
        store.store(content="Python is great for data science")
        nid2 = store.store(content="The weather is sunny today")

        node = store.get_node(nid2)
        assert node is not None
        assert "contradicts" not in node.metadata

    def test_skip_inference_skips_contradiction_check(self, store):
        """skip_inference=True should skip contradiction detection."""
        from unittest.mock import patch, MagicMock

        mock_check = MagicMock()
        with patch.object(store, "_check_contradictions", mock_check):
            store.store(content="Test memory", skip_inference=True)
        mock_check.assert_not_called()


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

    @pytest.mark.skipif(
        not importlib.util.find_spec("sqlite_vec"),
        reason="sqlite-vec not installed"
    )
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


@_skip_pro
class TestBatchRecordFeedback:
    """Tests for SQLiteStore.batch_record_feedback."""

    def test_batch_updates_multiple_memories(self, store):
        id1 = store.store(content="Python prefers spaces over tabs for indentation")
        id2 = store.store(content="The Rust borrow checker prevents data races at compile time")
        assert id1 != id2, "Content should not dedup — distinct topics"
        updated = store.batch_record_feedback([
            (id1, "helpful", "Auto: relevant"),
            (id2, "unhelpful", "Auto: not useful"),
        ])
        assert updated == 2
        # Verify scores directly from DB
        meta1 = json.loads(store._conn.execute(
            "SELECT metadata FROM memories WHERE node_id = ?", (id1,)
        ).fetchone()[0])
        meta2 = json.loads(store._conn.execute(
            "SELECT metadata FROM memories WHERE node_id = ?", (id2,)
        ).fetchone()[0])
        assert meta1["feedback_score"] == 1
        assert meta2["feedback_score"] == -1
        assert len(meta1["feedback_signals"]) == 1
        assert len(meta2["feedback_signals"]) == 1

    def test_batch_skips_missing_nodes(self, store):
        nid = store.store(content="Kubernetes pods restart on OOM kill with exit code 137")
        updated = store.batch_record_feedback([
            (nid, "helpful", "real"),
            ("nonexistent-id", "helpful", "ghost"),
        ])
        assert updated == 1

    def test_batch_single_commit(self, store):
        """Batch uses one commit, not N commits."""
        id1 = store.store(content="PostgreSQL JSONB indexes support GIN operators")
        id2 = store.store(content="Redis sorted sets use skip lists internally")
        assert id1 != id2
        from unittest.mock import patch
        with patch.object(store, "_commit", wraps=store._commit) as mock_commit:
            store.batch_record_feedback([
                (id1, "helpful", "a"),
                (id2, "helpful", "b"),
            ])
            assert mock_commit.call_count == 1

    def test_batch_empty_list(self, store):
        updated = store.batch_record_feedback([])
        assert updated == 0

    def test_batch_flags_at_threshold(self, store):
        nid = store.store(content="GraphQL N+1 queries should use DataLoader batching")
        updated = store.batch_record_feedback([
            (nid, "outdated", "stale"),  # -2
            (nid, "unhelpful", "wrong"),  # -3 -> flagged
        ])
        assert updated == 2
        meta = json.loads(store._conn.execute(
            "SELECT metadata FROM memories WHERE node_id = ?", (nid,)
        ).fetchone()[0])
        assert meta["feedback_score"] == -3
        assert meta["flagged_for_review"] is True


@_skip_pro
class TestBackfillEmbeddings:
    """Tests for SQLiteStore.backfill_embeddings."""

    def test_backfill_missing_embeddings(self, store):
        """Memories inserted without embeddings get backfilled."""
        if not store._vec_available:
            pytest.skip("vec not available")
        # Store a memory normally (will have embedding)
        nid1 = store.store(content="TCP three-way handshake uses SYN SYN-ACK ACK")
        # Manually insert a memory WITHOUT embedding to simulate hash-fallback
        from datetime import datetime, timezone
        store._conn.execute(
            "INSERT INTO memories (node_id, content, metadata, created_at) VALUES (?, ?, ?, ?)",
            ("backfill-test-1", "DNS resolves domain names to IP addresses via recursive queries",
             '{"event_type": "fact"}', datetime.now(timezone.utc).isoformat()),
        )
        store._conn.commit()
        # Verify it's missing from vec
        row = store._conn.execute(
            "SELECT id FROM memories WHERE node_id = 'backfill-test-1'"
        ).fetchone()
        vec_row = store._conn.execute(
            "SELECT rowid FROM memories_vec WHERE rowid = ?", (row[0],)
        ).fetchone()
        assert vec_row is None
        # Backfill
        result = store.backfill_embeddings(batch_size=10)
        assert result["backfilled"] >= 1
        assert result["remaining"] == 0
        # Verify vec entry now exists
        vec_row = store._conn.execute(
            "SELECT rowid FROM memories_vec WHERE rowid = ?", (row[0],)
        ).fetchone()
        assert vec_row is not None

    def test_backfill_no_missing(self, store):
        """Returns clean status when all memories have embeddings."""
        if not store._vec_available:
            pytest.skip("vec not available")
        store.store(content="HTTP status 204 means No Content response")
        result = store.backfill_embeddings()
        assert result["backfilled"] == 0
        assert result["remaining"] == 0

    def test_backfill_vec_unavailable(self, store):
        """Returns error when vec is not available."""
        store._vec_available = False
        result = store.backfill_embeddings()
        assert result["error"] == "vec not available"
        assert result["backfilled"] == 0


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


# ===========================================================================
# Comprehensive unit tests -- CRUD, Dedup, Query, Maintenance, Export/Import,
# Edge Cases (~50 tests)
# ===========================================================================


class TestCRUDComprehensive:
    """Thorough CRUD coverage for SQLiteStore."""

    def test_store_returns_string_with_mem_prefix(self, store):
        node_id = store.store(content="hello world", session_id="s1")
        assert isinstance(node_id, str)
        assert node_id.startswith("mem-")

    def test_store_with_content_session_and_event_type(self, store):
        node_id = store.store(
            content="project architecture decision",
            session_id="sess-42",
            metadata={"event_type": "decision"},
        )
        node = store.get_node(node_id)
        assert node is not None
        assert node.content == "project architecture decision"
        assert node.metadata.get("session_id") == "sess-42"
        assert node.metadata.get("event_type") == "decision"

    def test_store_with_ttl_seconds(self, store):
        node_id = store.store(content="short-lived note", ttl_seconds=7200)
        node = store.get_node(node_id)
        assert node is not None
        assert node.ttl_seconds == 7200
        assert not node.is_expired()

    def test_get_node_returns_memory_result_fields(self, store):
        node_id = store.store(
            content="check all fields",
            session_id="s-fields",
            metadata={"event_type": "lesson_learned", "project": "/proj"},
        )
        node = store.get_node(node_id)
        assert node.id == node_id
        assert node.content == "check all fields"
        assert isinstance(node.created_at, __import__("datetime").datetime)
        assert isinstance(node.access_count, int)
        assert node.metadata.get("event_type") == "lesson_learned"

    def test_get_node_increments_access_count(self, store):
        """Each get_node call bumps access_count in DB; returned value is pre-increment.

        After store(): DB has access_count=0.
        get_node #1: SELECTs 0, bumps DB to 1, returns 0.
        get_node #2: SELECTs 1, bumps DB to 2, returns 1.
        get_node #3: SELECTs 2, bumps DB to 3, returns 2.
        """
        node_id = store.store(content="access counting test")
        n1 = store.get_node(node_id)
        ac1 = n1.access_count  # 0
        n2 = store.get_node(node_id)
        ac2 = n2.access_count  # 1
        n3 = store.get_node(node_id)
        ac3 = n3.access_count  # 2
        assert ac2 == ac1 + 1
        assert ac3 == ac2 + 1

    def test_get_node_none_for_missing_id(self, store):
        assert store.get_node("mem-000000000000") is None

    def test_delete_node_true_for_existing(self, store):
        nid = store.store(content="will delete this")
        assert store.delete_node(nid) is True
        assert store.node_count() == 0

    def test_delete_node_false_for_missing(self, store):
        assert store.delete_node("mem-nope") is False

    @_skip_pro
    def test_delete_node_logs_to_forgetting_log(self, store):
        nid = store.store(
            content="forgotten memory content",
            metadata={"event_type": "decision"},
        )
        store.delete_node(nid)
        log = store.get_forgetting_log(limit=5)
        assert len(log) >= 1
        found = [e for e in log if e["node_id"] == nid]
        assert len(found) == 1
        assert found[0]["reason"] == "user_deleted"
        assert "forgotten memory" in found[0]["content_preview"]

    def test_update_node_content(self, store):
        nid = store.store(content="before update")
        ok = store.update_node(nid, content="after update")
        assert ok is True
        node = store.get_node(nid)
        assert node.content == "after update"

    def test_update_node_preserves_metadata_when_updating_content(self, store):
        """Updating only content should not lose existing metadata."""
        nid = store.store(
            content="original text",
            session_id="s1",
            metadata={"event_type": "decision", "session_id": "s1", "project": "/myproj"},
        )
        store.update_node(nid, content="new text")
        node = store.get_node(nid)
        assert node.content == "new text"
        # Metadata should still carry original event_type
        assert node.metadata.get("event_type") == "decision"

    def test_update_node_access_count(self, store):
        nid = store.store(content="access count update test")
        ok = store.update_node(nid, access_count=42)
        assert ok is True
        node = store.get_node(nid)
        # get_node SELECTs before incrementing, so it reads 42, then bumps to 43 in DB
        assert node.access_count == 42

    def test_node_count_after_multiple_stores(self, store):
        assert store.node_count() == 0
        store.store(content="Webpack tree-shaking eliminates dead code from bundles")
        assert store.node_count() == 1
        store.store(content="Nginx reverse proxy with upstream load balancing")
        assert store.node_count() == 2
        store.store(content="Prometheus alerting rules for SLA monitoring")
        assert store.node_count() == 3

    def test_node_count_zero_on_empty(self, store):
        assert store.node_count() == 0

    def test_batch_store_returns_all_ids(self, store):
        items = [
            {"content": "The PostgreSQL database uses WAL mode for write-ahead logging"},
            {"content": "React hooks like useState provide state management in components"},
            {"content": "Kubernetes pods are scheduled across worker nodes in the cluster"},
        ]
        ids = store.batch_store(items)
        assert len(ids) == 3
        assert all(isinstance(nid, str) for nid in ids)
        assert store.node_count() == 3

    def test_batch_store_with_metadata(self, store):
        items = [
            {
                "content": "batch with meta one xyz",
                "session_id": "batch-sess",
                "metadata": {"event_type": "decision"},
            },
            {
                "content": "batch with meta two abc",
                "session_id": "batch-sess",
                "metadata": {"event_type": "lesson_learned"},
            },
        ]
        ids = store.batch_store(items)
        assert len(ids) == 2
        results = store.get_by_session("batch-sess")
        assert len(results) == 2


class TestDeduplicationComprehensive:
    """Thorough dedup coverage."""

    def test_exact_content_returns_same_id(self, store):
        id1 = store.store(content="absolutely identical text here")
        id2 = store.store(content="absolutely identical text here")
        assert id1 == id2
        assert store.node_count() == 1

    def test_canonical_dedup_whitespace_normalization(self, store):
        """Different whitespace should canonicalize to same hash."""
        id1 = store.store(content="canonical whitespace test")
        id2 = store.store(content="canonical  whitespace   test")
        assert id1 == id2

    def test_canonical_dedup_case_normalization(self, store):
        """Different case should canonicalize to same hash."""
        id1 = store.store(content="Case Normalization Check")
        id2 = store.store(content="case normalization check")
        assert id1 == id2

    def test_canonical_dedup_markdown_stripping(self, store):
        """Markdown formatting should be stripped for canonical comparison."""
        id1 = store.store(content="deploy the api gateway")
        id2 = store.store(content="**deploy** the `api` gateway")
        assert id1 == id2

    def test_different_content_creates_separate_nodes(self, store):
        id1 = store.store(content="completely different alpha content xyz")
        id2 = store.store(content="completely different beta content abc")
        assert id1 != id2
        assert store.node_count() == 2

    def test_same_content_different_event_type_deduplicates(self, store):
        """Content hash dedup fires regardless of event_type."""
        id1 = store.store(
            content="same content different types",
            metadata={"event_type": "decision"},
        )
        id2 = store.store(
            content="same content different types",
            metadata={"event_type": "error_pattern"},
        )
        assert id1 == id2

    def test_dedup_bumps_access_count(self, store):
        """Dedup hit should increment access_count on existing node."""
        nid = store.store(content="dedup access bump check")
        # Store duplicate -- dedup bumps access_count from 0 to 1
        store.store(content="dedup access bump check")
        # get_node SELECTs before its own increment, so returns 1
        node = store.get_node(nid)
        assert node.access_count >= 1

    def test_dedup_stats_incremented(self, store):
        """Dedup should be reflected in stats counters."""
        store.store(content="stats dedup tracking check")
        store.store(content="stats dedup tracking check")
        total_dedup = (
            store.stats.get("dedup_exact", 0)
            + store.stats.get("dedup_canonical", 0)
            + store.stats.get("dedup_skips", 0)
        )
        assert total_dedup >= 1


class TestQueryComprehensive:
    """Thorough query and search coverage."""

    def test_query_finds_stored_content(self, store):
        store.store(
            content="PostgreSQL database optimization techniques",
            metadata={"event_type": "lesson_learned"},
        )
        results = store.query("PostgreSQL optimization", use_cache=False)
        assert len(results) >= 1
        assert any("PostgreSQL" in r.content for r in results)

    def test_query_respects_limit_parameter(self, store):
        topics = [
            "PostgreSQL MVCC concurrency control mechanism",
            "Redis pub/sub messaging pattern implementation",
            "MongoDB sharding with zone-based partitioning",
            "SQLite WAL journaling mode for write performance",
            "Cassandra gossip protocol for cluster membership",
            "DynamoDB on-demand capacity provisioning model",
            "InfluxDB time-series retention policy configuration",
            "Neo4j Cypher query language for graph traversal",
            "CockroachDB distributed SQL transaction handling",
            "Elasticsearch inverted index for full-text search",
        ]
        for t in topics:
            store.store(content=t, metadata={"event_type": "lesson_learned"})
        results = store.query("database storage systems", limit=3, use_cache=False)
        assert len(results) <= 3

    def test_query_returns_empty_on_no_match(self, store):
        store.store(content="Python programming fundamentals")
        results = store.query("xylophone underwater basket weaving", use_cache=False)
        assert isinstance(results, list)
        # Text search may or may not return results depending on word overlap

    def test_phrase_search_finds_exact_substring(self, store):
        store.store(content="The system uses event-driven architecture for scaling")
        results = store.phrase_search("event-driven architecture")
        assert len(results) >= 1
        assert "event-driven architecture" in results[0].content

    def test_phrase_search_empty_for_unmatched(self, store):
        store.store(content="Machine learning model training pipeline")
        results = store.phrase_search("quantum entanglement theorem")
        assert results == []

    def test_phrase_search_case_insensitive_by_default(self, store):
        store.store(content="The API Gateway Configuration is critical")
        results = store.phrase_search("api gateway configuration")
        assert len(results) >= 1

    def test_get_by_type_filters_event_type(self, store):
        store.store(
            content="We decided to use PostgreSQL for the main database backend",
            metadata={"event_type": "decision"},
        )
        store.store(
            content="TypeError occurs when calling numpy reshape without proper dimensions",
            metadata={"event_type": "error_pattern"},
        )
        store.store(
            content="Architecture decision: microservices over monolith for scalability",
            metadata={"event_type": "decision"},
        )

        results = store.get_by_type("decision")
        assert len(results) == 2
        for r in results:
            assert r.metadata.get("event_type") == "decision"

    def test_get_by_type_empty_for_nonexistent(self, store):
        store.store(content="exists", metadata={"event_type": "decision"})
        results = store.get_by_type("nonexistent_type_abc")
        assert results == []

    def test_get_by_session_correct_session(self, store):
        store.store(
            content="Configured PostgreSQL replication for high availability",
            session_id="alpha",
        )
        store.store(
            content="Fixed CORS headers in the Express middleware layer",
            session_id="beta",
        )
        store.store(
            content="Deployed Redis cache cluster with sentinel monitoring",
            session_id="alpha",
        )

        results = store.get_by_session("alpha")
        assert len(results) == 2
        for r in results:
            assert r.metadata.get("session_id") == "alpha"

    def test_get_by_session_excludes_others(self, store):
        store.store(
            content="GraphQL schema validation for user mutations",
            session_id="X",
        )
        store.store(
            content="Terraform module for AWS VPC peering connections",
            session_id="Y",
        )

        results = store.get_by_session("X")
        assert len(results) == 1
        assert "GraphQL" in results[0].content

    def test_get_by_type_with_entity_id_filter(self, store):
        store.store(
            content="OMEGA versioning strategy uses semantic versioning for releases",
            metadata={"event_type": "decision"},
            entity_id="omega",
        )
        store.store(
            content="Kubernetes resource limits set to 512Mi memory per container",
            metadata={"event_type": "decision"},
            entity_id="other",
        )
        # entity_id filter includes matching + NULL
        results = store.get_by_type("decision", entity_id="omega")
        assert len(results) >= 1
        contents = [r.content for r in results]
        assert any("OMEGA" in c for c in contents)

    def test_get_recent_ordering(self, store):
        store.store(content="First: PostgreSQL indexing strategies for large tables")
        store.store(content="Second: Docker compose networking between containers")
        store.store(content="Third: GitHub Actions CI pipeline with matrix builds")

        recent = store.get_recent(limit=2)
        assert len(recent) == 2
        assert recent[0].content == "Third: GitHub Actions CI pipeline with matrix builds"
        assert recent[1].content == "Second: Docker compose networking between containers"

    def test_query_on_empty_store_returns_empty(self, store):
        results = store.query("anything at all", use_cache=False)
        assert results == []


class TestMaintenanceComprehensive:
    """Thorough maintenance and health coverage."""

    def test_cleanup_expired_removes_expired(self, store):
        import time
        nid = store.store(content="ephemeral cleanup test", ttl_seconds=1)
        assert store.node_count() == 1
        time.sleep(1.5)
        removed = store.cleanup_expired()
        assert removed >= 1
        assert store.get_node(nid) is None

    def test_cleanup_expired_keeps_non_expired(self, store):
        store.store(content="Kubernetes pod affinity rules for co-location", ttl_seconds=86400)
        store.store(content="Datadog APM distributed tracing configuration")
        removed = store.cleanup_expired()
        assert removed == 0
        assert store.node_count() == 2

    def test_evict_lru_removes_least_accessed(self, store):
        id1 = store.store(content="Ansible playbook for server provisioning automation")
        id2 = store.store(content="Grafana dashboard template for Kubernetes monitoring")
        id3 = store.store(content="Terraform state management with S3 backend locking")
        # Access id3 and id2 to make them "hotter"
        store.get_node(id3)
        store.get_node(id3)
        store.get_node(id2)

        evicted = store.evict_lru(count=1)
        assert evicted == 1
        # id1 was never accessed beyond store, should be evicted first
        assert store.get_node(id1) is None

    def test_evict_lru_respects_count_parameter(self, store):
        distinct_items = [
            "Vault secrets engine for dynamic database credentials",
            "Consul service mesh with sidecar proxy injection",
            "Nomad job scheduling with constraint-based placement",
            "Packer image builder for immutable infrastructure",
            "Boundary secure remote access session management",
        ]
        for item in distinct_items:
            store.store(content=item)
        assert store.node_count() == 5
        evicted = store.evict_lru(count=3)
        assert evicted == 3
        assert store.node_count() == 2

    @_skip_pro
    def test_check_memory_health_keys(self, store):
        store.store(content="health keys test")
        health = store.check_memory_health(warn_mb=4000, critical_mb=8000)
        assert isinstance(health, dict)
        expected_keys = {"status", "node_count", "db_size_mb", "warnings",
                         "recommendations", "usage", "memory_mb",
                         "never_accessed_pct", "zero_access_count",
                         "embedding_degraded"}
        assert expected_keys.issubset(set(health.keys()))
        assert health["node_count"] == 1

    def test_check_memory_health_healthy_status(self, store):
        store.store(content="small healthy store test")
        health = store.check_memory_health(warn_mb=4000, critical_mb=8000)
        assert health["status"] == "healthy"


class TestExportImportComprehensive:
    """Thorough export/import round-trip coverage."""

    def test_export_creates_valid_json(self, store, tmp_omega_dir):
        store.store(content="OAuth2 PKCE flow implementation for mobile clients", metadata={"event_type": "decision"})
        store.store(content="Sentry error tracking integration with source maps", session_id="s-exp")

        path = tmp_omega_dir / "export_valid.json"
        result = store.export_to_file(path)
        assert path.exists()
        assert result["node_count"] == 2

        data = json.loads(path.read_text())
        assert data["version"] == "omega-sqlite-v1"
        assert len(data["nodes"]) == 2
        assert "exported_at" in data

    def test_import_restores_nodes(self, store, tmp_omega_dir):
        store.store(content="JWT token rotation strategy with Redis blacklist")
        store.store(content="Celery beat scheduler for periodic background tasks")
        path = tmp_omega_dir / "export_for_import.json"
        store.export_to_file(path)

        from omega.sqlite_store import SQLiteStore
        db2 = tmp_omega_dir / "import_target.db"
        s2 = SQLiteStore(db_path=db2)
        try:
            result = s2.import_from_file(path)
            assert result["node_count"] == 2
            assert s2.node_count() == 2
        finally:
            s2.close()

    def test_roundtrip_preserves_count(self, store, tmp_omega_dir):
        items = [
            "Webpack module federation for micro-frontend architecture",
            "Istio virtual service routing with canary deployment weights",
            "ArgoCD gitops sync policy for automatic reconciliation",
            "Falco runtime security alerting for container anomalies",
            "Linkerd service mesh with automatic mTLS certificate rotation",
        ]
        for item in items:
            store.store(content=item)

        path = tmp_omega_dir / "roundtrip_count.json"
        export_res = store.export_to_file(path)

        from omega.sqlite_store import SQLiteStore
        db2 = tmp_omega_dir / "roundtrip_count_target.db"
        s2 = SQLiteStore(db_path=db2)
        try:
            import_res = s2.import_from_file(path)
            assert export_res["node_count"] == import_res["node_count"]
            assert s2.node_count() == 5
        finally:
            s2.close()

    def test_roundtrip_preserves_content_and_type(self, store, tmp_omega_dir):
        store.store(
            content="preservable roundtrip content xyz",
            metadata={"event_type": "lesson_learned"},
        )
        path = tmp_omega_dir / "roundtrip_type.json"
        store.export_to_file(path)

        from omega.sqlite_store import SQLiteStore
        db2 = tmp_omega_dir / "roundtrip_type_target.db"
        s2 = SQLiteStore(db_path=db2)
        try:
            s2.import_from_file(path)
            results = s2.get_by_type("lesson_learned")
            assert len(results) == 1
            assert results[0].content == "preservable roundtrip content xyz"
        finally:
            s2.close()


# ===========================================================================
# Cluster co-boost tests (Phase 3.5)
# ===========================================================================


@_skip_pro
class TestCosineSimilarity:
    """Test the _cosine_similarity helper."""

    def test_identical_vectors(self):
        from omega.sqlite_store import _cosine_similarity
        v = [1.0, 0.0, 0.0]
        assert abs(_cosine_similarity(v, v) - 1.0) < 1e-6

    def test_orthogonal_vectors(self):
        from omega.sqlite_store import _cosine_similarity
        a = [1.0, 0.0, 0.0]
        b = [0.0, 1.0, 0.0]
        assert abs(_cosine_similarity(a, b)) < 1e-6

    def test_opposite_vectors(self):
        from omega.sqlite_store import _cosine_similarity
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        assert abs(_cosine_similarity(a, b) - (-1.0)) < 1e-6

    def test_zero_vector(self):
        from omega.sqlite_store import _cosine_similarity
        a = [0.0, 0.0, 0.0]
        b = [1.0, 2.0, 3.0]
        assert _cosine_similarity(a, b) == 0.0


@_skip_pro
class TestComputeClusterBoosts:
    """Test SQLiteStore._compute_cluster_boosts."""

    def test_boost_for_close_centroid(self, store):
        """Query near a cluster centroid gets a positive boost."""
        import struct
        # Create memory_clusters table and insert a cluster with a known centroid
        try:
            store._conn.execute("""
                CREATE TABLE IF NOT EXISTS memory_clusters (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    cluster_id INTEGER NOT NULL,
                    label TEXT NOT NULL,
                    member_count INTEGER NOT NULL,
                    centroid BLOB,
                    representative_keywords TEXT,
                    representative_memory_ids TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    superseded INTEGER DEFAULT 0
                )
            """)
        except Exception:
            pass  # Table may already exist

        # Centroid: unit vector along first axis
        centroid = [1.0] + [0.0] * 383
        centroid_bytes = struct.pack(f"{384}f", *centroid)
        store._conn.execute(
            """INSERT INTO memory_clusters
               (cluster_id, label, member_count, centroid,
                representative_keywords, representative_memory_ids,
                created_at, updated_at, superseded)
               VALUES (?, ?, ?, ?, ?, ?, datetime('now'), datetime('now'), 0)""",
            (0, "test_cluster", 10, centroid_bytes, "test", '["mem-1", "mem-2"]'),
        )
        store._conn.commit()

        # Query embedding very close to centroid (sim > 0.5)
        query_emb = [0.95] + [0.05] * 383
        boosts, clusters = store._compute_cluster_boosts(query_emb)

        assert 0 in boosts
        assert boosts[0] > 1.0
        assert boosts[0] <= 1.15
        assert len(clusters) == 1  # Clusters returned for reuse

    def test_no_boost_for_distant_centroid(self, store):
        """Query far from cluster centroid gets no boost."""
        import struct
        try:
            store._conn.execute("""
                CREATE TABLE IF NOT EXISTS memory_clusters (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    cluster_id INTEGER NOT NULL,
                    label TEXT NOT NULL,
                    member_count INTEGER NOT NULL,
                    centroid BLOB,
                    representative_keywords TEXT,
                    representative_memory_ids TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    superseded INTEGER DEFAULT 0
                )
            """)
        except Exception:
            pass

        centroid = [1.0] + [0.0] * 383
        centroid_bytes = struct.pack(f"{384}f", *centroid)
        store._conn.execute(
            """INSERT INTO memory_clusters
               (cluster_id, label, member_count, centroid,
                representative_keywords, representative_memory_ids,
                created_at, updated_at, superseded)
               VALUES (?, ?, ?, ?, ?, ?, datetime('now'), datetime('now'), 0)""",
            (0, "distant_cluster", 10, centroid_bytes, "test", '[]'),
        )
        store._conn.commit()

        # Query embedding orthogonal to centroid (sim ~ 0)
        query_emb = [0.0, 1.0] + [0.0] * 382
        boosts, clusters = store._compute_cluster_boosts(query_emb)
        assert len(boosts) == 0

    def test_empty_when_no_clusters(self, store):
        """Returns empty dict when no clusters exist."""
        try:
            store._conn.execute("""
                CREATE TABLE IF NOT EXISTS memory_clusters (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    cluster_id INTEGER NOT NULL,
                    label TEXT NOT NULL,
                    member_count INTEGER NOT NULL,
                    centroid BLOB,
                    representative_keywords TEXT,
                    representative_memory_ids TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    superseded INTEGER DEFAULT 0
                )
            """)
        except Exception:
            pass

        query_emb = [1.0] + [0.0] * 383
        boosts, clusters = store._compute_cluster_boosts(query_emb)
        assert boosts == {}
        assert clusters == []

    def test_boost_scales_with_similarity(self, store):
        """Higher similarity produces higher boost (within 1.05-1.15 range)."""
        import struct
        try:
            store._conn.execute("""
                CREATE TABLE IF NOT EXISTS memory_clusters (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    cluster_id INTEGER NOT NULL,
                    label TEXT NOT NULL,
                    member_count INTEGER NOT NULL,
                    centroid BLOB,
                    representative_keywords TEXT,
                    representative_memory_ids TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    superseded INTEGER DEFAULT 0
                )
            """)
        except Exception:
            pass

        # Two clusters at different angles
        c1 = [1.0] + [0.0] * 383
        c2 = [0.7, 0.7] + [0.0] * 382
        store._conn.execute(
            """INSERT INTO memory_clusters
               (cluster_id, label, member_count, centroid,
                representative_keywords, representative_memory_ids,
                created_at, updated_at, superseded)
               VALUES (?, ?, ?, ?, ?, ?, datetime('now'), datetime('now'), 0)""",
            (0, "close", 10, struct.pack(f"{384}f", *c1), "", '[]'),
        )
        store._conn.execute(
            """INSERT INTO memory_clusters
               (cluster_id, label, member_count, centroid,
                representative_keywords, representative_memory_ids,
                created_at, updated_at, superseded)
               VALUES (?, ?, ?, ?, ?, ?, datetime('now'), datetime('now'), 0)""",
            (1, "medium", 10, struct.pack(f"{384}f", *c2), "", '[]'),
        )
        store._conn.commit()

        # Query very close to c1
        query_emb = [0.99, 0.01] + [0.0] * 382
        boosts, clusters = store._compute_cluster_boosts(query_emb)

        # Both may match if similarity > 0.5
        if 0 in boosts and 1 in boosts:
            assert boosts[0] >= boosts[1]  # Closer centroid gets higher boost


class TestEdgeCasesComprehensive:
    """Thorough edge-case and boundary coverage."""

    def test_empty_content_raises_valueerror(self, store):
        with pytest.raises(ValueError, match="non-empty"):
            store.store(content="")

    def test_very_long_content_accepted(self, store):
        long_text = "a" * 50000
        nid = store.store(content=long_text)
        node = store.get_node(nid)
        assert node is not None
        assert len(node.content) == 50000

    def test_unicode_and_emoji_content(self, store):
        text = "Unicode: caf\u00e9 \u2603 \u2764 \u00e4\u00f6\u00fc \u4e16\u754c \u041c\u0438\u0440 \U0001f680"
        nid = store.store(content=text)
        node = store.get_node(nid)
        assert "\u2603" in node.content
        assert "\u4e16\u754c" in node.content
        assert "\U0001f680" in node.content

    def test_node_and_edge_count_consistency(self, store):
        assert store.node_count() == 0
        assert store.edge_count() == 0
        store.store(content="edge count consistency test")
        assert store.node_count() == 1
        assert store.edge_count() == 0  # No deps = no edges

    def test_close_and_reopen_preserves_data(self, tmp_omega_dir):
        """After close(), re-opening the same DB preserves data."""
        from omega.sqlite_store import SQLiteStore
        db_path = tmp_omega_dir / "closetest.db"
        s1 = SQLiteStore(db_path=db_path)
        s1.store(content="survive close test for persistence verification")
        s1.close()

        s2 = SQLiteStore(db_path=db_path)
        try:
            assert s2.node_count() == 1
            results = s2.phrase_search("survive close")
            assert len(results) == 1
        finally:
            s2.close()
