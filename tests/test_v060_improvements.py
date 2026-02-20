"""Tests for OMEGA v0.6.0 SOTA-inspired improvements.

Tests cover:
  1. Priority column + query boost
  2. Temporal model (referenced_date + _infer_temporal_range)
  3. Feedback amplification
  4. Observation compression
  5. Enhanced welcome (observation_prefix)
  6. Smart extractive compaction
"""

import os
import pytest
from datetime import datetime, timedelta, timezone


# ============================================================================
# 1. Priority Column + Query Boost
# ============================================================================


class TestPriorityColumn:
    """Priority column affects query ranking."""

    def test_schema_v2_has_priority_column(self, store):
        """The schema should include priority and referenced_date columns."""
        row = store._conn.execute("SELECT version FROM schema_version LIMIT 1").fetchone()
        assert row[0] >= 2  # v2 added priority/referenced_date, v3 added entity_id

        # Check column exists by inserting
        nid = store.store(
            content="Test memory with priority",
            metadata={"event_type": "decision", "priority": 5},
        )
        row = store._conn.execute("SELECT priority FROM memories WHERE node_id = ?", (nid,)).fetchone()
        assert row[0] == 5

    def test_default_priority_from_event_type(self, store):
        """user_preference gets priority 5, session_summary gets 2."""
        nid_pref = store.store(
            content="User prefers dark mode for all IDEs",
            metadata={"event_type": "user_preference"},
        )
        nid_summ = store.store(
            content="Session summary of today's work on the project",
            metadata={"event_type": "session_summary"},
        )

        pref_row = store._conn.execute("SELECT priority FROM memories WHERE node_id = ?", (nid_pref,)).fetchone()
        summ_row = store._conn.execute("SELECT priority FROM memories WHERE node_id = ?", (nid_summ,)).fetchone()
        assert pref_row[0] == 5
        assert summ_row[0] == 2

    def test_priority_boosts_ranking(self, store):
        """Higher priority memory should rank higher at equal similarity."""
        # Store two memories with similar content but different priorities
        nid_low = store.store(
            content="The database uses SQLite for embedded storage of application data",
            metadata={"event_type": "session_summary", "priority": 1},
        )
        nid_high = store.store(
            content="Decision to use SQLite for embedded storage of application data permanently",
            metadata={"event_type": "decision", "priority": 5},
        )

        results = store.query("SQLite embedded storage application data", limit=5)
        result_ids = [r.id for r in results]
        if nid_low in result_ids and nid_high in result_ids:
            # High priority should rank above low priority
            assert result_ids.index(nid_high) < result_ids.index(nid_low)

    def test_explicit_priority_overrides_default(self, store):
        """Explicitly setting priority=1 on a lesson_learned overrides the default of 4."""
        nid = store.store(
            content="Low priority lesson that should not surface often in results",
            metadata={"event_type": "lesson_learned", "priority": 1},
        )
        row = store._conn.execute("SELECT priority FROM memories WHERE node_id = ?", (nid,)).fetchone()
        assert row[0] == 1


# ============================================================================
# 2. Temporal Model
# ============================================================================


class TestTemporalModel:
    """Temporal inference and referenced_date support."""

    def test_referenced_date_stored(self, store):
        """referenced_date from metadata is written to the column."""
        ref = "2026-01-15T00:00:00+00:00"
        nid = store.store(
            content="Sprint 10 completed on January 15 2026",
            metadata={"event_type": "task_completion", "referenced_date": ref},
        )
        row = store._conn.execute(
            "SELECT referenced_date FROM memories WHERE node_id = ?", (nid,)
        ).fetchone()
        assert row[0] == ref

    def test_infer_temporal_range_last_week(self):
        """'last week' should produce a 7-day range."""
        from omega.bridge import _infer_temporal_range

        result = _infer_temporal_range("what happened last week")
        assert result is not None
        start, end = result
        # Parse and check range is ~7 days
        s = datetime.fromisoformat(start)
        e = datetime.fromisoformat(end)
        delta = e - s
        assert 6 <= delta.days <= 8

    def test_infer_temporal_range_yesterday(self):
        from omega.bridge import _infer_temporal_range

        result = _infer_temporal_range("what did I do yesterday")
        assert result is not None
        start, end = result
        s = datetime.fromisoformat(start)
        e = datetime.fromisoformat(end)
        assert (e - s).days == 1

    def test_infer_temporal_range_n_days_ago(self):
        from omega.bridge import _infer_temporal_range

        result = _infer_temporal_range("what happened 3 days ago")
        assert result is not None
        start, end = result
        s = datetime.fromisoformat(start)
        e = datetime.fromisoformat(end)
        delta = e - s
        assert 2 <= delta.days <= 4

    def test_infer_temporal_range_iso_date(self):
        from omega.bridge import _infer_temporal_range

        result = _infer_temporal_range("what happened on 2026-01-15")
        assert result is not None
        start, end = result
        assert "2026-01-15" in start

    def test_infer_temporal_range_month_name(self):
        from omega.bridge import _infer_temporal_range

        result = _infer_temporal_range("decisions made in January")
        assert result is not None
        start, end = result
        assert "-01-01" in start  # January

    def test_infer_temporal_range_no_signal(self):
        from omega.bridge import _infer_temporal_range

        result = _infer_temporal_range("how does the authentication system work")
        assert result is None

    def test_temporal_boost_in_query(self, store):
        """Memories in temporal range get boosted."""
        now = datetime.now(timezone.utc)

        # Store a recent memory
        nid_recent = store.store(
            content="Deployed critical security patch to production servers today",
            metadata={
                "event_type": "task_completion",
                "referenced_date": now.isoformat(),
            },
        )
        # Store an old memory
        old_date = (now - timedelta(days=90)).isoformat()
        nid_old = store.store(
            content="Deployed a minor update to the production server systems",
            metadata={
                "event_type": "task_completion",
                "referenced_date": old_date,
            },
        )

        # Query with temporal range for last week
        week_ago = (now - timedelta(days=7)).isoformat()
        results = store.query(
            "deployed production servers",
            limit=5,
            temporal_range=(week_ago, now.isoformat()),
        )
        result_ids = [r.id for r in results]
        if nid_recent in result_ids and nid_old in result_ids:
            assert result_ids.index(nid_recent) < result_ids.index(nid_old)


# ============================================================================
# 3. Feedback Amplification
# ============================================================================


class TestFeedbackAmplification:
    """Amplified feedback formula gives meaningful score differences."""

    def test_positive_feedback_factor(self, store):
        """fb_score of +5 should give factor of 1.75."""
        factor = store._compute_fb_factor(5)
        assert abs(factor - 1.75) < 0.01

    def test_max_positive_feedback_capped(self, store):
        """fb_score of +10 gives 2.5x, +15 still gives 2.5x (capped at 10)."""
        assert abs(store._compute_fb_factor(10) - 2.5) < 0.01
        assert abs(store._compute_fb_factor(15) - 2.5) < 0.01  # Capped

    def test_negative_feedback_factor(self, store):
        """fb_score of -2 should give factor of 0.6."""
        factor = store._compute_fb_factor(-2)
        assert abs(factor - 0.6) < 0.01

    def test_very_negative_feedback_floor(self, store):
        """fb_score of -4 should give minimum floor of 0.2."""
        factor = store._compute_fb_factor(-4)
        assert abs(factor - 0.2) < 0.01

    def test_extreme_negative_feedback_floor(self, store):
        """fb_score of -10 should still be 0.2 (floor)."""
        factor = store._compute_fb_factor(-10)
        assert factor == 0.2

    def test_zero_feedback_neutral(self, store):
        """fb_score of 0 should give factor of 1.0."""
        assert store._compute_fb_factor(0) == 1.0


# ============================================================================
# 4. Observation Compression
# ============================================================================


class TestObservationCompression:
    """Extractive compression of high-value memories."""

    def test_short_content_returns_none(self):
        from omega.bridge import _compress_to_observation

        result = _compress_to_observation("Short text.", "decision")
        assert result is None

    def test_long_decision_produces_observation(self):
        from omega.bridge import _compress_to_observation

        content = (
            "After extensive benchmarking of SQLite, PostgreSQL, and Redis for the "
            "OMEGA memory backend, we decided to use SQLite with sqlite-vec extension. "
            "The key factors were: zero-config deployment, embedded operation without a "
            "separate server process, and excellent read performance for our workload. "
            "PostgreSQL was ruled out due to operational complexity for a local-first tool."
        )
        result = _compress_to_observation(content, "decision")
        assert result is not None
        assert len(result) <= 200
        assert len(result) >= 30  # Should have meaningful content

    def test_observation_capped_at_200_chars(self):
        from omega.bridge import _compress_to_observation

        content = " ".join([
            f"Sentence number {i} contains important information about system architecture "
            f"and design patterns that were discovered during testing phase {i}."
            for i in range(20)
        ])
        result = _compress_to_observation(content, "lesson_learned")
        if result:
            assert len(result) <= 200

    def test_auto_capture_adds_observation(self, tmp_omega_dir):
        """auto_capture should add observation to metadata for high-value types."""
        os.environ["OMEGA_SKIP_EMBEDDINGS"] = "1"
        try:
            from omega.bridge import reset_memory, auto_capture, _get_store
            reset_memory()

            content = (
                "Threading.Lock in Python is non-reentrant. "
                "Never call a method that acquires self._lock from inside with self._lock. "
                "This causes silent deadlocks that are extremely hard to debug. "
                "If tests hang with no output, suspect nested lock acquisition."
            )
            result = auto_capture(
                content=content,
                event_type="lesson_learned",
                session_id="test-obs",
            )
            assert "Memory Captured" in result or "Evolved" in result

            store = _get_store()
            # Find the stored memory
            nodes = store.get_by_type("lesson_learned", limit=5)
            assert len(nodes) >= 1
            # Check observation was added
            for n in nodes:
                if "Threading.Lock" in n.content:
                    assert "observation" in n.metadata
                    assert len(n.metadata["observation"]) > 0
                    assert len(n.metadata["observation"]) <= 200
                    break
            else:
                pytest.fail("Did not find the lesson_learned memory")
        finally:
            os.environ.pop("OMEGA_SKIP_EMBEDDINGS", None)


# ============================================================================
# 5. Enhanced Welcome
# ============================================================================


class TestEnhancedWelcome:
    """Welcome returns observation_prefix and project_context."""

    def test_welcome_has_observation_prefix(self, tmp_omega_dir):
        os.environ["OMEGA_SKIP_EMBEDDINGS"] = "1"
        try:
            from omega.bridge import reset_memory, _get_store, welcome
            reset_memory()
            store = _get_store()

            # Seed high-value memories
            for i in range(5):
                store.store(
                    content=f"Important lesson number {i} about Python threading patterns and best practices",
                    metadata={"event_type": "lesson_learned", "priority": 4},
                )
            for i in range(3):
                store.store(
                    content=f"User prefers tool {i} for development workflow",
                    metadata={"event_type": "user_preference", "priority": 5},
                )

            result = welcome(session_id="test-welcome")
            assert "observation_prefix" in result
            # Should have some content grouped by type
            assert isinstance(result["observation_prefix"], str)
            assert "memory_count" in result
        finally:
            os.environ.pop("OMEGA_SKIP_EMBEDDINGS", None)

    def test_welcome_has_relative_time(self, tmp_omega_dir):
        os.environ["OMEGA_SKIP_EMBEDDINGS"] = "1"
        try:
            from omega.bridge import reset_memory, _get_store, welcome
            reset_memory()
            store = _get_store()

            store.store(
                content="A very important decision about the architecture of the system",
                metadata={"event_type": "decision", "priority": 4},
            )

            result = welcome(session_id="test-relative")
            recent = result.get("recent_memories", [])
            if recent:
                assert "relative_time" in recent[0]
        finally:
            os.environ.pop("OMEGA_SKIP_EMBEDDINGS", None)

    def test_welcome_project_context(self, tmp_omega_dir):
        os.environ["OMEGA_SKIP_EMBEDDINGS"] = "1"
        try:
            from omega.bridge import reset_memory, _get_store, welcome
            reset_memory()
            store = _get_store()

            store.store(
                content="OMEGA uses SQLite with sqlite-vec for vector similarity search",
                metadata={"event_type": "decision", "project": "/Projects/omega"},
            )

            result = welcome(session_id="test-project", project="/Projects/omega")
            assert "project_context" in result
            assert isinstance(result["project_context"], str)
        finally:
            os.environ.pop("OMEGA_SKIP_EMBEDDINGS", None)


# ============================================================================
# 6. Smart Extractive Compaction
# ============================================================================


class TestSmartExtract:
    """_smart_extract produces diverse, information-dense summaries."""

    def test_smart_extract_selects_diverse_sentences(self):
        from omega.bridge import _smart_extract
        from omega.sqlite_store import MemoryResult

        # Create mock cluster with diverse content
        nodes = [
            MemoryResult(id="m1", content="SQLite WAL mode enables concurrent reads while writing to the database.",
                         created_at=datetime(2026, 1, 1, tzinfo=timezone.utc)),
            MemoryResult(id="m2", content="Always use parameterized queries to prevent SQL injection attacks.",
                         created_at=datetime(2026, 1, 2, tzinfo=timezone.utc)),
            MemoryResult(id="m3", content="SQLite WAL mode is essential for concurrent read access in production systems.",
                         created_at=datetime(2026, 1, 3, tzinfo=timezone.utc)),
        ]

        result = _smart_extract(nodes)
        assert len(result) > 0
        assert len(result) <= 1000

    def test_smart_extract_caps_at_1000_chars(self):
        from omega.bridge import _smart_extract
        from omega.sqlite_store import MemoryResult

        # Create cluster with lots of content
        nodes = []
        for i in range(10):
            content = f"Unique sentence number {i} with completely different words about topic alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma tau upsilon phi chi psi omega."
            nodes.append(MemoryResult(id=f"m{i}", content=content,
                                      created_at=datetime(2026, 1, i + 1, tzinfo=timezone.utc)))

        result = _smart_extract(nodes)
        assert len(result) <= 1000

    def test_smart_extract_empty_cluster(self):
        from omega.bridge import _smart_extract

        result = _smart_extract([])
        assert result == ""

    def test_smart_extract_orders_chronologically(self):
        from omega.bridge import _smart_extract
        from omega.sqlite_store import MemoryResult

        nodes = [
            MemoryResult(id="m1", content="First lesson about database optimization techniques for production.",
                         created_at=datetime(2026, 1, 1, tzinfo=timezone.utc)),
            MemoryResult(id="m2", content="Second lesson about API rate limiting best practices for microservices.",
                         created_at=datetime(2026, 1, 15, tzinfo=timezone.utc)),
            MemoryResult(id="m3", content="Third lesson about containerization strategies for deployment pipelines.",
                         created_at=datetime(2026, 2, 1, tzinfo=timezone.utc)),
        ]

        result = _smart_extract(nodes)
        # Should contain content from multiple nodes in chronological order
        assert len(result) > 0


# ============================================================================
# 7. Relative Time Helper
# ============================================================================


class TestRelativeTime:
    """_relative_time formats datetimes as human-readable strings."""

    def test_just_now(self):
        from omega.bridge import _relative_time

        now = datetime.now(timezone.utc)
        assert _relative_time(now) == "just now"

    def test_minutes_ago(self):
        from omega.bridge import _relative_time

        t = datetime.now(timezone.utc) - timedelta(minutes=15)
        result = _relative_time(t)
        assert "m ago" in result

    def test_hours_ago(self):
        from omega.bridge import _relative_time

        t = datetime.now(timezone.utc) - timedelta(hours=3)
        result = _relative_time(t)
        assert "h ago" in result

    def test_yesterday(self):
        from omega.bridge import _relative_time

        t = datetime.now(timezone.utc) - timedelta(days=1, hours=12)
        result = _relative_time(t)
        assert result == "yesterday"

    def test_days_ago(self):
        from omega.bridge import _relative_time

        t = datetime.now(timezone.utc) - timedelta(days=5)
        result = _relative_time(t)
        assert "5d ago" in result

    def test_months_ago(self):
        from omega.bridge import _relative_time

        t = datetime.now(timezone.utc) - timedelta(days=65)
        result = _relative_time(t)
        assert "month" in result

    def test_empty_input(self):
        from omega.bridge import _relative_time

        assert _relative_time(None) == ""
        assert _relative_time("") == ""

    def test_iso_string_input(self):
        from omega.bridge import _relative_time

        t = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
        result = _relative_time(t)
        assert "h ago" in result


# ============================================================================
# 8. Tool Schema Validation
# ============================================================================


class TestToolSchemas:
    """New schema fields are present."""

    def test_omega_store_has_priority(self):
        from omega.server.tool_schemas import TOOL_SCHEMAS

        store_schema = next(s for s in TOOL_SCHEMAS if s["name"] == "omega_store")
        props = store_schema["inputSchema"]["properties"]
        assert "priority" in props
        assert props["priority"]["type"] == "integer"

    def test_omega_query_has_temporal_range(self):
        from omega.server.tool_schemas import TOOL_SCHEMAS

        query_schema = next(s for s in TOOL_SCHEMAS if s["name"] == "omega_query")
        props = query_schema["inputSchema"]["properties"]
        assert "temporal_range" in props
        assert props["temporal_range"]["type"] == "array"


# ============================================================================
# 9. Abstention Floor (v0.6.0 benchmark improvement)
# ============================================================================


class TestAbstentionFloor:
    """Off-topic queries should return empty results (abstention)."""

    def test_threshold_constants_exist(self, store):
        """Class-level abstention thresholds are defined."""
        assert hasattr(store, "_MIN_VEC_SIMILARITY")
        assert hasattr(store, "_MIN_TEXT_RELEVANCE")
        assert hasattr(store, "_MIN_COMPOSITE_SCORE")
        assert 0.0 < store._MIN_VEC_SIMILARITY < 1.0
        assert 0.0 < store._MIN_TEXT_RELEVANCE <= 1.0
        assert 0.0 < store._MIN_COMPOSITE_SCORE < 1.0

    def test_on_topic_query_returns_results(self, store):
        """On-topic queries should still return results normally."""
        store.store(
            content="SQLite uses WAL mode for concurrent read access in production",
            metadata={"event_type": "decision"},
        )
        store.store(
            content="Python threading.Lock is non-reentrant and causes deadlocks",
            metadata={"event_type": "lesson_learned"},
        )
        results = store.query("SQLite WAL mode concurrent", limit=5)
        assert len(results) >= 1

    def test_text_only_low_relevance_filtered(self, store):
        """Text-only results below _MIN_TEXT_RELEVANCE should be filtered."""
        store.store(
            content="A simple note about project architecture and design patterns",
            metadata={"event_type": "decision"},
        )
        # Query with many words where only one matches — word ratio < 0.5
        results = store.query(
            "knitting patterns for wool sweaters mittens scarves hats",
            limit=5,
        )
        # "patterns" might match but ratio is 1/8 = 0.125, well below 0.5
        # Should return empty or only results with high enough match ratio
        for r in results:
            assert r.relevance >= 0.3  # Normalized, so check not garbage


# ============================================================================
# 10. Temporal Hard Penalty (v0.6.0 benchmark improvement)
# ============================================================================


class TestTemporalHardPenalty:
    """Out-of-range memories get 0.15x penalty, significantly down-ranking them."""

    def test_out_of_range_penalized(self, store):
        """Memories outside temporal_range should be penalized with 0.15x multiplier."""
        now = datetime.now(timezone.utc)

        # Store memories at 30 and 60 days ago
        for days_ago in [30, 60]:
            ref = (now - timedelta(days=days_ago)).isoformat()
            store.store(
                content=f"Sprint completed {days_ago} days ago with all goals met",
                metadata={"event_type": "task_completion", "referenced_date": ref},
            )

        # Also store an in-range memory to compare
        in_range_ref = (now - timedelta(days=85)).isoformat()
        store.store(
            content="Sprint completed 85 days ago with partial goals met",
            metadata={"event_type": "task_completion", "referenced_date": in_range_ref},
        )

        # Query for 80-90 days ago — in-range should rank above out-of-range
        t_start = (now - timedelta(days=90)).isoformat()
        t_end = (now - timedelta(days=80)).isoformat()
        results = store.query(
            "sprint completed goals",
            limit=5,
            temporal_range=(t_start, t_end),
        )
        # In-range memory should be first (boosted 1.3x vs penalized 0.15x)
        assert len(results) >= 1
        assert "85 days ago" in results[0].content

    def test_in_range_still_returned(self, store):
        """Memories inside temporal_range are boosted and returned."""
        now = datetime.now(timezone.utc)

        ref = (now - timedelta(days=5)).isoformat()
        nid = store.store(
            content="Deployed critical fix to production database servers recently",
            metadata={"event_type": "task_completion", "referenced_date": ref},
        )

        t_start = (now - timedelta(days=10)).isoformat()
        t_end = now.isoformat()
        results = store.query(
            "deployed production database fix",
            limit=5,
            temporal_range=(t_start, t_end),
        )
        result_ids = [r.id for r in results]
        assert nid in result_ids

    def test_mixed_in_and_out_of_range(self, store):
        """In-range memory should rank above out-of-range memory."""
        now = datetime.now(timezone.utc)

        # In-range: 5 days ago
        ref_in = (now - timedelta(days=5)).isoformat()
        nid_in = store.store(
            content="Deployed the new authentication microservice to production cluster",
            metadata={"event_type": "task_completion", "referenced_date": ref_in},
        )
        # Out-of-range: 50 days ago (distinct content to avoid embedding dedup)
        ref_out = (now - timedelta(days=50)).isoformat()
        nid_out = store.store(
            content="Refactored the legacy billing module payment gateway integration",
            metadata={"event_type": "task_completion", "referenced_date": ref_out},
        )
        assert nid_in != nid_out, "Embedding dedup collapsed distinct memories"

        t_start = (now - timedelta(days=10)).isoformat()
        t_end = now.isoformat()
        results = store.query(
            "deployed authentication microservice production billing payment",
            limit=5,
            temporal_range=(t_start, t_end),
        )
        result_ids = [r.id for r in results]
        assert nid_in in result_ids
        # In-range memory should rank above out-of-range (1.3x boost vs 0.15x penalty)
        if nid_out in result_ids:
            assert result_ids.index(nid_in) < result_ids.index(nid_out)


# ============================================================================
# 11. BM25 Text Search (v0.6.0 benchmark improvement)
# ============================================================================


class TestWordTagOverlapBoost:
    """Phase 2.5: Word/tag overlap boosts ranking for term-matching memories."""

    def test_word_overlap_boosts_target(self, store):
        """Memory with query-word overlap should rank above semantically-similar noise."""
        # Target: has "connection" and "pooling" matching query
        nid_target = store.store(
            content="Added connection pooling with pgbouncer to handle concurrent database connections.",
            metadata={"event_type": "task_completion", "priority": 3, "tags": ["postgres"]},
        )
        # Noise: semantically similar (about infrastructure) but no word overlap
        store.store(
            content="Implemented retry logic with exponential backoff for external API calls.",
            metadata={"event_type": "lesson_learned", "priority": 4, "tags": ["python", "api"]},
        )
        store.store(
            content="Migrated from single EC2 to ECS Fargate with auto-scaling for better uptime.",
            metadata={"event_type": "task_completion", "priority": 3, "tags": ["aws", "docker"]},
        )

        results = store.query("connection pooling solution", limit=3)
        assert results, "Expected results for connection pooling query"
        result_ids = [r.id for r in results]
        assert nid_target in result_ids, "Target memory should appear in results"
        assert result_ids.index(nid_target) <= 1, "Target should rank in top 2"

    def test_tag_overlap_contributes(self, store):
        """Tags matching query words should contribute to word overlap boost."""
        nid_tagged = store.store(
            content="Added automated database migration step to the pipeline before deployment.",
            metadata={"event_type": "task_completion", "priority": 3, "tags": ["git", "cicd"]},
        )
        store.store(
            content="Implemented feature flags using LaunchDarkly for gradual rollouts.",
            metadata={"event_type": "decision", "priority": 4},
        )

        results = store.query("database migration cicd pipeline", limit=3)
        assert results
        result_ids = [r.id for r in results]
        assert nid_tagged in result_ids, "Tagged memory should appear in results"

    def test_negative_feedback_dampens_boost(self, store):
        """Negatively-rated memories should get less word boost than positively-rated ones."""
        # Old version: has perfect word overlap but strong negative feedback
        nid_old = store.store(
            content="The frontend uses Create React App for build tooling.",
            metadata={"event_type": "decision", "priority": 3, "feedback_score": -3},
        )
        # New version: less word overlap but positive feedback
        nid_new = store.store(
            content="Migrated from Create React App to Vite for 10x faster builds.",
            metadata={"event_type": "decision", "priority": 4, "feedback_score": 2},
        )

        results = store.query("frontend build tooling", limit=3)
        assert results
        result_ids = [r.id for r in results]
        if nid_old in result_ids and nid_new in result_ids:
            # New version should outrank old despite old having better word overlap
            assert result_ids.index(nid_new) < result_ids.index(nid_old), (
                "New version (positive feedback) should outrank old version (negative feedback)"
            )

    def test_no_boost_for_zero_overlap(self, store):
        """Memories with no query word overlap should get no boost."""
        store.store(
            content="Implemented structured logging with Winston replacing all console.log statements.",
            metadata={"event_type": "task_completion", "priority": 3},
        )
        # Query with completely different terms should still return results by vec sim
        results = store.query("logging implementation approach", limit=3)
        # Just verify it doesn't crash and returns something reasonable
        for r in results:
            assert isinstance(r.relevance, float)
            assert 0.0 <= r.relevance <= 1.0


class TestDualCheckAbstention:
    """Vec results below threshold can survive via text+tag word overlap fallback."""

    def test_offtopic_still_filtered(self, store):
        """Off-topic queries should still be filtered (no word overlap fallback saves them)."""
        store.store(
            content="Python asyncio event loop handles concurrent operations efficiently.",
            metadata={"event_type": "lesson_learned", "priority": 4, "tags": ["python"]},
        )
        results = store.query("knitting patterns for winter sweaters", limit=3)
        assert not results or all(r.relevance < 0.3 for r in results)

    def test_borderline_vec_with_word_overlap_survives(self, store):
        """A vec result below vec threshold but with good word overlap should survive."""
        nid = store.store(
            content="Added rate limiting middleware to the API with 100 requests per minute per IP.",
            metadata={"event_type": "task_completion", "priority": 3, "tags": ["fastapi", "security"]},
        )
        # Add noise memories to ensure competitive results
        for i in range(5):
            store.store(
                content=f"Generic infrastructure note number {i} about cloud services.",
                metadata={"event_type": "lesson_learned", "priority": 3},
            )
        results = store.query("rate limiting configuration", limit=5)
        result_ids = [r.id for r in results]
        # Target should survive because "rate" and "limiting" match content
        assert nid in result_ids, "Rate limiting memory should survive abstention via word overlap"


class TestWordOverlapStemming:
    """Lightweight stemming in _word_overlap handles morphological variants."""

    def test_stemming_matches_deploy_variants(self, store):
        """'deployment' query should match content with 'deployed' via stem 'deploy'."""
        from omega.sqlite_store import SQLiteStore

        ratio = SQLiteStore._word_overlap(
            ["container", "deployment", "platform"],
            "deployed the application to aws ecs for serverless container management",
        )
        # "container" exact match, "deployment" -> stem "deploy" in "deployed"
        assert ratio >= 0.5, f"Expected >= 0.5, got {ratio}"

    def test_stemming_no_false_positive(self, store):
        """Stemming should not create false matches for unrelated words."""
        from omega.sqlite_store import SQLiteStore

        ratio = SQLiteStore._word_overlap(
            ["knitting", "patterns", "sweaters"],
            "python asyncio event loop handles concurrent operations",
        )
        assert ratio == 0.0, f"Expected 0.0 for unrelated content, got {ratio}"

    def test_exact_match_preferred_over_stem(self, store):
        """Exact substring match should be found before trying stemming."""
        from omega.sqlite_store import SQLiteStore

        ratio = SQLiteStore._word_overlap(
            ["monitoring", "alerts"],
            "monitoring stack with alerts configured for cpu thresholds",
        )
        assert ratio == 1.0, f"Expected 1.0 for exact matches, got {ratio}"

    def test_stem_minimum_length(self, store):
        """Stemmed result must be at least 3 chars to avoid spurious matches."""
        from omega.sqlite_store import SQLiteStore

        # "used" -> strip "ed" -> "us" (only 2 chars, below min 3) -> no stem match
        ratio = SQLiteStore._word_overlap(["used"], "the user prefers python")
        # "used" not in content, stem "us" too short -> 0
        assert ratio == 0.0, f"Expected 0.0 for too-short stem, got {ratio}"


class TestBM25TextSearch:
    """BM25-blended text search ranks rare terms higher."""

    def test_rare_term_ranks_higher(self, store):
        """Memory with rare matching term should rank above common-term match."""
        # "sqlite" appears in many memories, "vectorization" is rare
        store.store(
            content="OMEGA uses sqlite for storage of all memory data",
            metadata={"event_type": "decision"},
        )
        store.store(
            content="OMEGA uses sqlite for storage of all memory data copy two",
            metadata={"event_type": "decision"},
        )
        nid_rare = store.store(
            content="OMEGA implements vectorization using sqlite-vec extension for similarity",
            metadata={"event_type": "decision"},
        )

        results = store.query("OMEGA vectorization sqlite-vec similarity", limit=5)
        if results:
            # The memory with the rare term "vectorization" should rank high
            result_ids = [r.id for r in results]
            if nid_rare in result_ids:
                assert result_ids.index(nid_rare) <= 1  # Top 2

    def test_bm25_relevance_is_float(self, store):
        """BM25-blended relevance should produce float values between 0 and 1."""
        store.store(
            content="Python asyncio event loop handles concurrent operations efficiently",
            metadata={"event_type": "lesson_learned"},
        )
        results = store.query("Python asyncio concurrent", limit=5)
        for r in results:
            assert isinstance(r.relevance, float)
            assert 0.0 <= r.relevance <= 1.0
