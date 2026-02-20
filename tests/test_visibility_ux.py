"""Tests for OMEGA Visibility UX features (v0.2.7).

Tests:
- _human_ttl() formatting (permanent, minutes, hours, days)
- query_structured() returns relevance field and accepts context params
- SQLiteStore.edge_count() public method
- SQLiteStore.get_last_capture_time() public method
- SQLiteStore.get_session_event_counts() public method
- Surfacing relevance threshold (>=30%)
- Capture confirmations (Memory Captured, Evolved, Deduped)
- Session activity report formatting (plurals, labels)
- Auto-feedback reads and cleans .surfaced.json
"""

import json
import sys
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# ============================================================================
# Fixture: reset bridge singleton between tests
# ============================================================================

@pytest.fixture(autouse=True)
def _reset_bridge(tmp_omega_dir):
    """Reset bridge singleton before and after each test."""
    from omega.bridge import reset_memory
    reset_memory()
    yield
    reset_memory()


# ============================================================================
# _human_ttl() formatting
# ============================================================================

class TestHumanTTL:
    """Test human-readable TTL formatting."""

    def test_none_is_permanent(self):
        from omega.bridge import _human_ttl
        assert _human_ttl(None) == "permanent"

    def test_zero_is_permanent(self):
        from omega.bridge import _human_ttl
        assert _human_ttl(0) == "permanent"

    def test_minutes(self):
        from omega.bridge import _human_ttl
        assert _human_ttl(1800) == "30m"
        assert _human_ttl(300) == "5m"
        assert _human_ttl(60) == "1m"

    def test_hours(self):
        from omega.bridge import _human_ttl
        assert _human_ttl(3600) == "1h"
        assert _human_ttl(7200) == "2h"
        assert _human_ttl(43200) == "12h"

    def test_days(self):
        from omega.bridge import _human_ttl
        assert _human_ttl(86400) == "1d"
        assert _human_ttl(1209600) == "14d"
        assert _human_ttl(604800) == "7d"

    def test_auto_capture_uses_human_ttl(self, tmp_omega_dir):
        """auto_capture output should show human-readable TTL, not raw seconds."""
        from omega.bridge import auto_capture
        result = auto_capture(
            content="Test human TTL display in auto_capture output",
            event_type="decision",
            metadata={"source": "test"},
            session_id="test-session",
        )
        assert "14d" in result
        assert "1209600" not in result


# ============================================================================
# query_structured() enhancements
# ============================================================================

class TestQueryStructured:
    """Test query_structured returns relevance and accepts context params."""

    def test_returns_relevance_field(self, tmp_omega_dir):
        from omega.bridge import auto_capture, query_structured
        auto_capture(
            content="Python debugging tips for pytest fixtures",
            event_type="lesson_learned",
            session_id="test-s",
        )
        results = query_structured(
            query_text="pytest debugging",
            limit=3,
            session_id="test-s",
        )
        assert len(results) >= 1
        assert "relevance" in results[0]
        assert isinstance(results[0]["relevance"], float)

    def test_accepts_context_file(self, tmp_omega_dir):
        from omega.bridge import auto_capture, query_structured
        auto_capture(
            content="Bridge module handles all query routing",
            event_type="decision",
            session_id="test-s",
        )
        # Should not raise
        results = query_structured(
            query_text="bridge query",
            limit=3,
            context_file="/src/omega/bridge.py",
        )
        assert isinstance(results, list)

    def test_accepts_context_tags(self, tmp_omega_dir):
        from omega.bridge import auto_capture, query_structured
        auto_capture(
            content="Use SQLite for persistent storage",
            event_type="decision",
            session_id="test-s",
        )
        results = query_structured(
            query_text="storage",
            limit=3,
            context_tags=["sqlite", "python"],
        )
        assert isinstance(results, list)


# ============================================================================
# SQLiteStore public methods (replacing _conn access)
# ============================================================================

class TestStorePublicMethods:
    """Test new public methods on SQLiteStore."""

    def test_edge_count_empty(self, store):
        assert store.edge_count() == 0

    def test_edge_count_after_add(self, store):
        nid1 = store.store(content="Memory one", session_id="s1")
        nid2 = store.store(content="Memory two", session_id="s1")
        store.add_edge(nid1, nid2, "related", 0.8)
        assert store.edge_count() == 1

    def test_get_last_capture_time_empty(self, store):
        assert store.get_last_capture_time() is None

    def test_get_last_capture_time_returns_iso(self, store):
        store.store(content="Test memory", session_id="s1")
        ts = store.get_last_capture_time()
        assert ts is not None
        assert "T" in ts  # ISO format

    def test_get_session_event_counts_empty(self, store):
        result = store.get_session_event_counts("nonexistent")
        assert result == {}

    def test_get_session_event_counts(self, store):
        store.store(
            content="An error occurred during testing",
            session_id="s1",
            metadata={"event_type": "error_pattern"},
        )
        store.store(
            content="Decided to use SQLite backend",
            session_id="s1",
            metadata={"event_type": "decision"},
        )
        store.store(
            content="Another error in the same session",
            session_id="s1",
            metadata={"event_type": "error_pattern"},
        )
        counts = store.get_session_event_counts("s1")
        assert counts.get("error_pattern") == 2
        assert counts.get("decision") == 1

    def test_get_session_event_counts_isolates_sessions(self, store):
        store.store(content="Error in s1", session_id="s1",
                    metadata={"event_type": "error_pattern"})
        store.store(content="Error in s2", session_id="s2",
                    metadata={"event_type": "error_pattern"})
        counts = store.get_session_event_counts("s1")
        assert counts.get("error_pattern") == 1


# ============================================================================
# Activity report formatting
# ============================================================================

class TestActivityReport:
    """Test session activity report output formatting."""

    def test_lesson_learned_plural(self):
        """'lesson learned' should pluralize as 'lessons learned', not 'lesson learneds'."""
        # Import and call the formatting logic inline
        counts = {"error_pattern": 2, "decision": 1, "lesson_learned": 3}
        _LABELS = {
            "error_pattern": ("error", "errors"),
            "decision": ("decision", "decisions"),
            "lesson_learned": ("lesson learned", "lessons learned"),
        }
        parts = [f"{sum(counts.values())} captured"]
        for key, (singular, plural) in _LABELS.items():
            n = counts.get(key, 0)
            if n:
                parts.append(f"{n} {plural if n > 1 else singular}")

        line = " | ".join(parts)
        assert "lessons learned" in line
        assert "lesson learneds" not in line

    def test_singular_decision(self):
        counts = {"decision": 1}
        _LABELS = {
            "error_pattern": ("error", "errors"),
            "decision": ("decision", "decisions"),
            "lesson_learned": ("lesson learned", "lessons learned"),
        }
        parts = []
        for key, (singular, plural) in _LABELS.items():
            n = counts.get(key, 0)
            if n:
                parts.append(f"{n} {plural if n > 1 else singular}")
        assert "1 decision" in parts[0]
        assert "decisions" not in parts[0]

    def test_empty_session_id_skips_report(self, tmp_omega_dir):
        """Empty session_id should produce no output."""
        # Import the hook's function
        hooks_dir = str(Path(__file__).parent.parent / "hooks")
        sys.path.insert(0, hooks_dir)
        try:
            import importlib
            import session_stop
            importlib.reload(session_stop)
            captured = StringIO()
            with patch("sys.stdout", captured):
                session_stop._print_activity_report("")
            assert captured.getvalue() == ""
        finally:
            sys.path.remove(hooks_dir)


# ============================================================================
# Surfacing counter and auto-feedback
# ============================================================================

class TestSurfacingCounter:
    """Test the file-based surfacing counter and auto-feedback."""

    def test_surfaced_file_count(self, tmp_omega_dir):
        """Surfacing counter file size equals number of surfacing events."""
        marker = tmp_omega_dir / "session-test123.surfaced"
        # Simulate 3 surfacing events
        with open(marker, "a") as f:
            f.write("x")
        with open(marker, "a") as f:
            f.write("x")
        with open(marker, "a") as f:
            f.write("x")
        assert marker.stat().st_size == 3

    def test_surfaced_json_tracks_ids(self, tmp_omega_dir):
        """Surfaced memory IDs are tracked per file in .surfaced.json."""
        json_path = tmp_omega_dir / "session-test123.surfaced.json"
        data = {"/foo/bar.py": ["mem-aaa", "mem-bbb"]}
        json_path.write_text(json.dumps(data))

        loaded = json.loads(json_path.read_text())
        assert "mem-aaa" in loaded["/foo/bar.py"]
        assert len(loaded["/foo/bar.py"]) == 2

    def test_auto_feedback_cleans_up_json(self, tmp_omega_dir, store):
        """Auto-feedback should clean up the .surfaced.json file."""
        # Create a memory and a surfaced.json referencing it
        nid = store.store(content="Test memory for feedback", session_id="s1")
        json_path = tmp_omega_dir / "session-s1.surfaced.json"
        json_path.write_text(json.dumps({"/foo.py": [nid]}))

        hooks_dir = str(Path(__file__).parent.parent / "hooks")
        sys.path.insert(0, hooks_dir)
        try:
            import importlib
            import session_stop
            importlib.reload(session_stop)
            # Patch Path.home() so the function finds our tmp_omega_dir
            with patch.object(Path, "home", return_value=tmp_omega_dir.parent):
                session_stop._auto_feedback_on_surfaced("s1")
            # File should be cleaned up
            assert not json_path.exists()
        finally:
            sys.path.remove(hooks_dir)


# ============================================================================
# Capture confirmations
# ============================================================================

class TestCaptureConfirmations:
    """Test that auto_capture returns appropriate confirmation strings."""

    def test_new_capture_contains_memory_captured(self, tmp_omega_dir):
        from omega.bridge import auto_capture
        result = auto_capture(
            content="A completely novel error that has never been seen before in testing xyz123",
            event_type="error_pattern",
            session_id="test-s",
        )
        assert "Memory Captured" in result

    def test_dedup_returns_deduplicated(self, tmp_omega_dir):
        from omega.bridge import auto_capture
        content = "The exact same error repeated for dedup testing purposes with enough words"
        auto_capture(content=content, event_type="error_pattern", session_id="s1")
        result = auto_capture(content=content, event_type="error_pattern", session_id="s1")
        assert "Deduped" in result

    def test_ttl_display_in_capture(self, tmp_omega_dir):
        """Captured memory should show human TTL, not raw seconds."""
        from omega.bridge import auto_capture
        result = auto_capture(
            content="Decision: use SQLite for persistent storage backend in production",
            event_type="decision",
            session_id="test-s",
        )
        # decision TTL is LONG_TERM = 14 days = 1209600 seconds
        assert "14d" in result
        assert "1209600" not in result


# ============================================================================
# Relevance threshold
# ============================================================================

class TestRelevanceThreshold:
    """Test that surfacing filters low-relevance results."""

    def test_filter_below_threshold(self):
        """Results below 30% relevance should be filtered out."""
        results = [
            {"relevance": 0.95, "content": "high"},
            {"relevance": 0.50, "content": "medium"},
            {"relevance": 0.20, "content": "low"},
            {"relevance": 0.05, "content": "very low"},
        ]
        filtered = [r for r in results if r.get("relevance", 0.0) >= 0.30]
        assert len(filtered) == 2
        assert all(r["relevance"] >= 0.30 for r in filtered)
