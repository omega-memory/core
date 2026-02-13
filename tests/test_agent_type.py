"""Tests for agent_type scoping — sub-agent memory isolation."""

from omega.sqlite_store import SQLiteStore, SCHEMA_VERSION


# ============================================================================
# Schema Tests
# ============================================================================


class TestAgentTypeSchema:
    """Test schema v4 migration and agent_type column."""

    def test_schema_version_is_4(self, store):
        assert SCHEMA_VERSION == 5
        row = store._conn.execute("SELECT version FROM schema_version LIMIT 1").fetchone()
        assert row[0] == 5

    def test_agent_type_column_exists(self, store):
        """The memories table should have an agent_type column."""
        info = store._conn.execute("PRAGMA table_info(memories)").fetchall()
        col_names = [col[1] for col in info]
        assert "agent_type" in col_names

    def test_agent_type_index_exists(self, store):
        """An index on agent_type should exist."""
        indexes = store._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_memories_agent_type'"
        ).fetchall()
        assert len(indexes) == 1

    def test_migration_v3_to_v4(self, tmp_omega_dir):
        """Simulate a v3 database and verify migration adds agent_type."""
        db_path = tmp_omega_dir / "migrate_test.db"
        import sqlite3

        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL)")
        conn.execute("INSERT INTO schema_version (version) VALUES (3)")
        conn.execute("""
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
                entity_id TEXT
            )
        """)
        conn.commit()
        conn.close()

        # Opening SQLiteStore should trigger migration
        s = SQLiteStore(db_path=db_path)
        info = s._conn.execute("PRAGMA table_info(memories)").fetchall()
        col_names = [col[1] for col in info]
        assert "agent_type" in col_names

        row = s._conn.execute("SELECT version FROM schema_version LIMIT 1").fetchone()
        assert row[0] == 5  # v3 -> v4 (agent_type) + v4 -> v5 (canonical_hash)
        s.close()


# ============================================================================
# Store Tests
# ============================================================================


class TestAgentTypeStore:
    """Test storing memories with agent_type."""

    def test_store_with_agent_type(self, store):
        """Store with explicit agent_type param."""
        node_id = store.store(
            content="Code review: always check null pointers",
            session_id="s1",
            metadata={"event_type": "lesson_learned"},
            agent_type="code-reviewer",
        )
        assert node_id.startswith("mem-")

        row = store._conn.execute(
            "SELECT agent_type FROM memories WHERE node_id = ?", (node_id,)
        ).fetchone()
        assert row[0] == "code-reviewer"

    def test_store_without_agent_type(self, store):
        """Store without agent_type — column should be NULL."""
        node_id = store.store(
            content="Generic memory without agent scope",
            session_id="s1",
        )

        row = store._conn.execute(
            "SELECT agent_type FROM memories WHERE node_id = ?", (node_id,)
        ).fetchone()
        assert row[0] is None

    def test_store_infer_agent_type_from_metadata(self, store):
        """agent_type in metadata should be used when param is not passed."""
        node_id = store.store(
            content="Test runner pattern: use fixtures for DB setup",
            session_id="s1",
            metadata={"event_type": "lesson_learned", "agent_type": "test-runner"},
        )

        row = store._conn.execute(
            "SELECT agent_type FROM memories WHERE node_id = ?", (node_id,)
        ).fetchone()
        assert row[0] == "test-runner"

    def test_store_param_overrides_metadata(self, store):
        """Explicit param takes precedence over metadata."""
        node_id = store.store(
            content="Override test",
            session_id="s1",
            metadata={"agent_type": "from-metadata"},
            agent_type="from-param",
        )

        row = store._conn.execute(
            "SELECT agent_type FROM memories WHERE node_id = ?", (node_id,)
        ).fetchone()
        assert row[0] == "from-param"


# ============================================================================
# Query Tests
# ============================================================================


class TestAgentTypeQuery:
    """Test querying with agent_type filter."""

    def _seed(self, store):
        """Seed memories with different agent types."""
        store.store(
            content="Review pattern: check for SQL injection in user inputs",
            session_id="s1",
            metadata={"event_type": "lesson_learned"},
            agent_type="code-reviewer",
            skip_inference=True,
        )
        store.store(
            content="Test pattern: use parameterized fixtures for database tests",
            session_id="s1",
            metadata={"event_type": "lesson_learned"},
            agent_type="test-runner",
            skip_inference=True,
        )
        store.store(
            content="General memory: SQL injection prevention is important",
            session_id="s1",
            metadata={"event_type": "lesson_learned"},
            skip_inference=True,
        )

    def test_query_filters_by_agent_type(self, store):
        """Query with agent_type should only return matching memories."""
        self._seed(store)

        results = store.query("SQL injection", limit=10, agent_type="code-reviewer")
        assert len(results) >= 1
        for r in results:
            # All results should have agent_type=code-reviewer
            row = store._conn.execute(
                "SELECT agent_type FROM memories WHERE node_id = ?", (r.id,)
            ).fetchone()
            assert row[0] == "code-reviewer"

    def test_query_without_agent_type_returns_all(self, store):
        """Query without agent_type should return all matching memories."""
        self._seed(store)

        results = store.query("pattern", limit=10)
        # Should include memories from all agent types + unscoped
        assert len(results) >= 2

    def test_query_agent_type_no_match(self, store):
        """Query with non-existent agent_type returns empty."""
        self._seed(store)

        results = store.query("SQL injection", limit=10, agent_type="nonexistent-agent")
        assert len(results) == 0


# ============================================================================
# Bridge Tests
# ============================================================================


class TestAgentTypeBridge:
    """Test bridge layer passes agent_type through."""

    def test_bridge_store_with_agent_type(self, tmp_omega_dir):
        """bridge.store() passes agent_type to SQLiteStore."""
        from omega.bridge import store, _get_store

        result = store(
            content="Bridge test: code reviewer lesson on error handling patterns",
            event_type="lesson_learned",
            agent_type="code-reviewer",
        )
        assert "Memory Captured" in result or "Memory Evolved" in result or "Memory Blocked" not in result

        db = _get_store()
        # Find the stored memory
        rows = db._conn.execute(
            "SELECT agent_type FROM memories WHERE agent_type = 'code-reviewer'"
        ).fetchall()
        assert len(rows) >= 1
        assert rows[0][0] == "code-reviewer"

    def test_bridge_store_agent_type_in_metadata(self, tmp_omega_dir):
        """bridge.store() should include agent_type in metadata."""
        from omega.bridge import store, _get_store

        store(
            content="Bridge metadata test: agent type should appear in metadata for this lesson",
            event_type="lesson_learned",
            agent_type="test-runner",
        )

        db = _get_store()
        import json
        rows = db._conn.execute(
            "SELECT metadata FROM memories WHERE agent_type = 'test-runner'"
        ).fetchall()
        assert len(rows) >= 1
        meta = json.loads(rows[0][0])
        assert meta.get("agent_type") == "test-runner"

    def test_bridge_query_with_agent_type(self, tmp_omega_dir):
        """bridge.query() passes agent_type filter through."""
        from omega.bridge import store, query

        store(
            content="Agent-scoped lesson: always validate input boundaries in code review",
            event_type="lesson_learned",
            agent_type="code-reviewer",
        )
        store(
            content="Agent-scoped lesson: always validate test fixtures before running",
            event_type="lesson_learned",
            agent_type="test-runner",
        )

        # Query scoped to code-reviewer
        result = query(query_text="validate", agent_type="code-reviewer")
        assert "Query Results" in result


# ============================================================================
# Lessons Tests
# ============================================================================


class TestAgentTypeLessons:
    """Test lessons filtering by agent_type."""

    def test_cross_session_lessons_filtered(self, tmp_omega_dir):
        """get_cross_session_lessons() filters by agent_type."""
        from omega.bridge import store, get_cross_session_lessons

        store(
            content="Code review lesson: always check error handling paths thoroughly",
            event_type="lesson_learned",
            session_id="s1",
            agent_type="code-reviewer",
        )
        store(
            content="Test runner lesson: use fixture isolation for parallel test execution",
            event_type="lesson_learned",
            session_id="s1",
            agent_type="test-runner",
        )
        store(
            content="General lesson: document all breaking changes in changelog",
            event_type="lesson_learned",
            session_id="s1",
        )

        # Filter to code-reviewer
        lessons = get_cross_session_lessons(agent_type="code-reviewer", limit=10)
        for lesson in lessons:
            assert "code review" in lesson["content"].lower() or "error handling" in lesson["content"].lower()

        # Filter to test-runner
        lessons = get_cross_session_lessons(agent_type="test-runner", limit=10)
        for lesson in lessons:
            assert "fixture" in lesson["content"].lower() or "test" in lesson["content"].lower()

    def test_cross_project_lessons_filtered(self, tmp_omega_dir):
        """get_cross_project_lessons() filters by agent_type."""
        from omega.bridge import store, get_cross_project_lessons

        store(
            content="Cross-project code review lesson: enforce consistent naming conventions",
            event_type="lesson_learned",
            session_id="s1",
            agent_type="code-reviewer",
            metadata={"project": "/proj/a"},
        )
        store(
            content="Cross-project test lesson: always seed deterministic test data",
            event_type="lesson_learned",
            session_id="s1",
            agent_type="test-runner",
            metadata={"project": "/proj/b"},
        )

        lessons = get_cross_project_lessons(agent_type="code-reviewer", limit=10)
        for lesson in lessons:
            assert "code review" in lesson["content"].lower() or "naming" in lesson["content"].lower()

    def test_lessons_no_agent_type_returns_all(self, tmp_omega_dir):
        """Without agent_type, lessons returns all types."""
        from omega.bridge import store, get_cross_session_lessons

        store(
            content="Lesson A: agent-scoped reviewer insight on code quality standards",
            event_type="lesson_learned",
            session_id="s1",
            agent_type="code-reviewer",
        )
        store(
            content="Lesson B: unscoped general insight on debugging methodology",
            event_type="lesson_learned",
            session_id="s1",
        )

        lessons = get_cross_session_lessons(limit=10)
        # Should include both scoped and unscoped
        assert len(lessons) >= 2
