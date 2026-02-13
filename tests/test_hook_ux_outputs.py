"""Tests for hook UX output formatting.

Covers:
- Scored surfacing output format ([score%] event_type: preview (id:xxx))
- Health pulse formatting (ago calculation, edge count, label)
- Activity report via actual hook function (_print_activity_report)
- Build summary from session_stop
- _ext_to_tags full extension coverage
- _surface_for_edit output format with scoring
- Cross-project lesson output format in standalone hook
"""

import json
import os
import sys
from io import StringIO
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "hooks"))


# ============================================================================
# Fixture: reset bridge singleton between tests
# ============================================================================

@pytest.fixture(autouse=True)
def _reset_bridge(tmp_omega_dir):
    from omega.bridge import reset_memory
    reset_memory()
    yield
    reset_memory()


@pytest.fixture
def fake_home(tmp_omega_dir):
    home_dir = tmp_omega_dir.parent
    with patch.object(Path, "home", return_value=home_dir):
        yield home_dir


# ============================================================================
# Scored surfacing output format
# ============================================================================

class TestScoredSurfacingFormat:
    """Test _surface_for_edit output lines."""

    def test_header_contains_filename(self):
        from omega.server.hook_server import _surface_for_edit
        mock_results = [
            {"relevance": 0.87, "event_type": "decision", "content": "Use SQLite", "id": "abcdef1234"},
        ]
        with patch("omega.bridge.query_structured", return_value=mock_results):
            lines = _surface_for_edit("/src/omega/bridge.py", "s1", "/Projects/omega")
        assert any("[MEMORY]" in line and "bridge.py" in line for line in lines)

    def test_score_percentage_format(self):
        from omega.server.hook_server import _surface_for_edit
        mock_results = [
            {"relevance": 0.87, "event_type": "decision", "content": "Use SQLite", "id": "abcdef1234"},
        ]
        with patch("omega.bridge.query_structured", return_value=mock_results):
            lines = _surface_for_edit("/src/bridge.py", "s1", "/proj")
        scored_lines = [line for line in lines if "%" in line]
        assert len(scored_lines) == 1
        assert "[87%]" in scored_lines[0]

    def test_event_type_in_output(self):
        from omega.server.hook_server import _surface_for_edit
        mock_results = [
            {"relevance": 0.65, "event_type": "error_pattern", "content": "DB timeout", "id": "xyz789abcd"},
        ]
        with patch("omega.bridge.query_structured", return_value=mock_results):
            lines = _surface_for_edit("/f.py", "s1", "/p")
        scored_lines = [line for line in lines if "%" in line]
        assert "error_pattern:" in scored_lines[0]

    def test_memory_id_truncated_to_8_chars(self):
        from omega.server.hook_server import _surface_for_edit
        full_id = "abcdef1234567890"
        mock_results = [
            {"relevance": 0.50, "event_type": "decision", "content": "test", "id": full_id},
        ]
        with patch("omega.bridge.query_structured", return_value=mock_results):
            lines = _surface_for_edit("/f.py", "s1", "/p")
        scored_lines = [line for line in lines if "id:" in line]
        assert f"(id:{full_id[:8]})" in scored_lines[0]
        assert full_id not in scored_lines[0]

    def test_content_preview_truncated_at_120(self):
        from omega.server.hook_server import _surface_for_edit
        long_content = "A" * 200
        mock_results = [
            {"relevance": 0.40, "event_type": "decision", "content": long_content, "id": "abc12345"},
        ]
        with patch("omega.bridge.query_structured", return_value=mock_results):
            lines = _surface_for_edit("/f.py", "s1", "/p")
        scored_lines = [line for line in lines if "%" in line]
        # Line should contain at most 120 A's worth of content
        assert "A" * 121 not in scored_lines[0]

    def test_newlines_in_content_replaced_with_spaces(self):
        from omega.server.hook_server import _surface_for_edit
        mock_results = [
            {"relevance": 0.55, "event_type": "decision", "content": "line1\nline2\nline3", "id": "abc12345"},
        ]
        with patch("omega.bridge.query_structured", return_value=mock_results):
            lines = _surface_for_edit("/f.py", "s1", "/p")
        scored_lines = [line for line in lines if "%" in line]
        assert "\n" not in scored_lines[0].split("[55%]")[1]  # after score, no newlines
        assert "line1 line2 line3" in scored_lines[0]

    def test_results_below_threshold_filtered(self):
        from omega.server.hook_server import _surface_for_edit
        mock_results = [
            {"relevance": 0.10, "event_type": "decision", "content": "low", "id": "aaa"},
            {"relevance": 0.05, "event_type": "error_pattern", "content": "very low", "id": "bbb"},
        ]
        with patch("omega.bridge.query_structured", return_value=mock_results):
            lines = _surface_for_edit("/f.py", "s1", "/p")
        assert lines == []

    def test_empty_results_returns_empty(self):
        from omega.server.hook_server import _surface_for_edit
        with patch("omega.bridge.query_structured", return_value=[]):
            lines = _surface_for_edit("/f.py", "s1", "/p")
        assert lines == []

    def test_multiple_results_all_formatted(self):
        from omega.server.hook_server import _surface_for_edit
        mock_results = [
            {"relevance": 0.90, "event_type": "decision", "content": "First", "id": "aaa11111"},
            {"relevance": 0.75, "event_type": "lesson_learned", "content": "Second", "id": "bbb22222"},
            {"relevance": 0.60, "event_type": "error_pattern", "content": "Third", "id": "ccc33333"},
        ]
        with patch("omega.bridge.query_structured", return_value=mock_results):
            lines = _surface_for_edit("/f.py", "s1", "/p")
        # 1 header + 3 scored lines
        assert len(lines) == 4
        assert "[90%]" in lines[1]
        assert "[75%]" in lines[2]
        assert "[60%]" in lines[3]

    def test_tracks_surfaced_ids(self, fake_home, tmp_omega_dir):
        from omega.server.hook_server import _surface_for_edit
        mock_results = [
            {"relevance": 0.80, "event_type": "decision", "content": "test", "id": "mem-aaa"},
        ]
        with patch("omega.bridge.query_structured", return_value=mock_results):
            _surface_for_edit("/f.py", "s1", "/p")
        json_path = tmp_omega_dir / "session-s1.surfaced.json"
        assert json_path.exists()
        data = json.loads(json_path.read_text())
        assert "mem-aaa" in data["/f.py"]


# ============================================================================
# Health pulse formatting
# ============================================================================

class TestHealthPulse:
    """Test health pulse output in session_start hook."""

    def _run_main_capture(self, fake_home, tmp_omega_dir, **overrides):
        """Run session_start.main() capturing stdout."""
        import importlib
        import session_start
        importlib.reload(session_start)

        mock_welcome = {
            "greeting": "Welcome back!",
            "memory_count": 42,
            "recent_memories": [],
        }
        mock_health = overrides.get("health", {"ok": True})
        edge_count = overrides.get("edge_count", 100)
        last_ts = overrides.get("last_ts", None)

        mock_store = MagicMock()
        mock_store.edge_count.return_value = edge_count
        mock_store.count.return_value = overrides.get("node_count", 50)
        mock_store.get_last_capture_time.return_value = last_ts

        captured = StringIO()
        with patch("omega.bridge.welcome", return_value=mock_welcome), \
             patch("omega.bridge.status", return_value=mock_health), \
             patch("omega.bridge._get_store", return_value=mock_store), \
             patch("omega.bridge.consolidate"), \
             patch("omega.bridge.compact"), \
             patch("omega.bridge.get_cross_project_lessons", return_value=[]), \
             patch("sys.stdout", captured):
            session_start.main()
        return captured.getvalue()

    def test_health_label_ok(self, fake_home, tmp_omega_dir):
        output = self._run_main_capture(fake_home, tmp_omega_dir, health={"ok": True})
        assert "**Health:** ok" in output

    def test_health_label_from_status(self, fake_home, tmp_omega_dir):
        output = self._run_main_capture(fake_home, tmp_omega_dir, health={"ok": False, "status": "degraded"})
        assert "**Health:** degraded" in output

    def test_edge_count_thousands_separator(self, fake_home, tmp_omega_dir):
        output = self._run_main_capture(fake_home, tmp_omega_dir, edge_count=12345)
        assert "12,345 edges" in output

    def test_last_capture_never(self, fake_home, tmp_omega_dir):
        output = self._run_main_capture(fake_home, tmp_omega_dir, last_ts=None)
        assert "**Last capture:** never" in output

    def test_last_capture_seconds_ago(self, fake_home, tmp_omega_dir):
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc)
        ts = now.isoformat()
        output = self._run_main_capture(fake_home, tmp_omega_dir, last_ts=ts)
        assert "s ago" in output

    def test_last_capture_days_ago(self, fake_home, tmp_omega_dir):
        from datetime import datetime, timezone, timedelta
        old = datetime.now(timezone.utc) - timedelta(days=3, hours=5)
        ts = old.isoformat()
        output = self._run_main_capture(fake_home, tmp_omega_dir, last_ts=ts)
        assert "3d ago" in output

    def test_full_health_line_format(self, fake_home, tmp_omega_dir):
        output = self._run_main_capture(fake_home, tmp_omega_dir, edge_count=50)
        assert "**Health:**" in output
        assert "**Graph:**" in output
        assert "**Last capture:**" in output


# ============================================================================
# Activity report via actual hook function
# ============================================================================

class TestActivityReportHook:
    """Test _print_activity_report from the actual session_stop hook."""

    def _capture_report(self, session_id, counts, surfaced=0, fake_home=None, tmp_omega_dir=None):
        import importlib
        import session_stop
        importlib.reload(session_stop)

        mock_store = MagicMock()
        mock_store.get_session_event_counts.return_value = counts

        if tmp_omega_dir and surfaced:
            marker = tmp_omega_dir / f"session-{session_id}.surfaced"
            marker.write_text("x" * surfaced)

        captured = StringIO()
        with patch("omega.bridge._get_store", return_value=mock_store), \
             patch("sys.stdout", captured):
            session_stop._print_activity_report(session_id)
        return captured.getvalue()

    def test_full_report_format(self, fake_home, tmp_omega_dir):
        counts = {"error_pattern": 2, "decision": 3, "lesson_learned": 1}
        output = self._capture_report("s1", counts, surfaced=4, fake_home=fake_home, tmp_omega_dir=tmp_omega_dir)
        assert "Session complete" in output
        assert "6 captured" in output  # 2+3+1
        assert "2 errors" in output
        assert "3 decisions" in output
        assert "1 lesson learned" in output
        assert "4 surfaced" in output

    def test_pipe_delimited(self, fake_home, tmp_omega_dir):
        counts = {"decision": 2}
        output = self._capture_report("s1", counts, fake_home=fake_home, tmp_omega_dir=tmp_omega_dir)
        assert "|" in output

    def test_no_output_when_empty(self, fake_home, tmp_omega_dir):
        output = self._capture_report("s1", {}, fake_home=fake_home, tmp_omega_dir=tmp_omega_dir)
        assert output == ""

    def test_no_output_for_empty_session_id(self, fake_home, tmp_omega_dir):
        output = self._capture_report("", {"decision": 1}, fake_home=fake_home, tmp_omega_dir=tmp_omega_dir)
        assert output == ""

    def test_surfaced_only(self, fake_home, tmp_omega_dir):
        """Report should appear even with zero captured but some surfaced."""
        output = self._capture_report("s1", {}, surfaced=5, fake_home=fake_home, tmp_omega_dir=tmp_omega_dir)
        assert "0 captured" in output
        assert "5 surfaced" in output

    def test_plural_errors(self, fake_home, tmp_omega_dir):
        counts = {"error_pattern": 3}
        output = self._capture_report("s1", counts, fake_home=fake_home, tmp_omega_dir=tmp_omega_dir)
        assert "3 errors" in output

    def test_singular_error(self, fake_home, tmp_omega_dir):
        counts = {"error_pattern": 1}
        output = self._capture_report("s1", counts, fake_home=fake_home, tmp_omega_dir=tmp_omega_dir)
        assert "1 error" in output
        assert "errors" not in output


# ============================================================================
# Build summary from session_stop
# ============================================================================

class TestBuildSummary:
    """Test _build_summary in the session_stop hook."""

    def _build(self, decisions=None, errors=None, tasks=None):
        import importlib
        import session_stop
        importlib.reload(session_stop)

        def fake_query(query_text, limit, session_id, project, event_type=None):
            if event_type == "decision":
                return decisions or []
            elif event_type == "error_pattern":
                return errors or []
            elif event_type == "task_completion":
                return tasks or []
            return []

        with patch("omega.bridge.query_structured", side_effect=fake_query):
            return session_stop._build_summary("s1", "/proj")

    def test_no_activity_fallback(self):
        result = self._build()
        assert result == "Session ended (no captured activity)"

    def test_decisions_section(self):
        result = self._build(decisions=[
            {"content": "Used SQLite for storage"},
            {"content": "Chose ONNX over PyTorch"},
        ])
        assert "Decisions (2)" in result
        assert "Used SQLite" in result
        assert "Chose ONNX" in result

    def test_errors_section(self):
        result = self._build(errors=[
            {"content": "DB connection timeout on cold start"},
        ])
        assert "Errors (1)" in result
        assert "DB connection timeout" in result

    def test_tasks_section(self):
        result = self._build(tasks=[
            {"content": "Migrated store to SQLite"},
        ])
        assert "Tasks (1)" in result
        assert "Migrated store" in result

    def test_multiple_sections_pipe_delimited(self):
        result = self._build(
            decisions=[{"content": "d1"}],
            errors=[{"content": "e1"}],
        )
        assert " | " in result

    def test_summary_truncated_at_600(self):
        long_decisions = [{"content": "A" * 120} for _ in range(10)]
        result = self._build(decisions=long_decisions)
        assert len(result) <= 600

    def test_content_preview_truncated_at_120(self):
        result = self._build(decisions=[{"content": "B" * 200}])
        # Decision content should be truncated
        assert "B" * 121 not in result


# ============================================================================
# _ext_to_tags full coverage
# ============================================================================

class TestExtToTagsComplete:
    """Test all 18 file extension mappings."""

    @pytest.mark.parametrize("ext,expected", [
        (".py", ["python"]),
        (".js", ["javascript"]),
        (".ts", ["typescript"]),
        (".tsx", ["typescript", "react"]),
        (".jsx", ["javascript", "react"]),
        (".rs", ["rust"]),
        (".go", ["go"]),
        (".rb", ["ruby"]),
        (".java", ["java"]),
        (".swift", ["swift"]),
        (".sh", ["bash"]),
        (".sql", ["sql"]),
        (".md", ["markdown"]),
        (".yml", ["yaml"]),
        (".yaml", ["yaml"]),
        (".json", ["json"]),
        (".toml", ["toml"]),
    ])
    def test_extension_mapping(self, ext, expected):
        from omega.server.hook_server import _ext_to_tags
        assert _ext_to_tags(f"/tmp/file{ext}") == expected

    def test_case_insensitive(self):
        from omega.server.hook_server import _ext_to_tags
        assert _ext_to_tags("/tmp/File.PY") == ["python"]
        assert _ext_to_tags("/tmp/App.TSX") == ["typescript", "react"]

    def test_no_extension(self):
        from omega.server.hook_server import _ext_to_tags
        assert _ext_to_tags("/tmp/Makefile") == []

    def test_unknown_extension(self):
        from omega.server.hook_server import _ext_to_tags
        assert _ext_to_tags("/tmp/data.csv") == []


# ============================================================================
# Cross-project lesson output format (standalone hook)
# ============================================================================

class TestCrossProjectOutputFormat:
    """Test the cross-project lesson output in session_start standalone hook."""

    def test_output_header(self, fake_home, tmp_omega_dir, capsys):
        import importlib
        import session_start
        importlib.reload(session_start)

        mock_lessons = [
            {"content": "Always validate inputs", "cross_project": True, "project": "other-project"},
        ]
        mock_welcome = {"greeting": "Hello", "memory_count": 10, "recent_memories": []}

        with patch("omega.bridge.welcome", return_value=mock_welcome), \
             patch("omega.bridge.consolidate"), \
             patch("omega.bridge.compact"), \
             patch("omega.bridge.get_cross_project_lessons", return_value=mock_lessons):
            session_start.main()

        output = capsys.readouterr().out
        assert "[CROSS-PROJECT]" in output
        assert "Lessons from other codebases:" in output

    def test_project_name_in_brackets(self, fake_home, tmp_omega_dir, capsys):
        import importlib
        import session_start
        importlib.reload(session_start)

        mock_lessons = [
            {"content": "Use type hints", "cross_project": True, "project": "/projects/acme"},
        ]
        mock_welcome = {"greeting": "Hello", "memory_count": 10, "recent_memories": []}

        with patch("omega.bridge.welcome", return_value=mock_welcome), \
             patch("omega.bridge.consolidate"), \
             patch("omega.bridge.compact"), \
             patch("omega.bridge.get_cross_project_lessons", return_value=mock_lessons):
            session_start.main()

        output = capsys.readouterr().out
        assert "[/projects/acme]" in output
        assert "Use type hints" in output

    def test_content_truncated_at_120(self, fake_home, tmp_omega_dir, capsys):
        import importlib
        import session_start
        importlib.reload(session_start)

        long_content = "X" * 200
        mock_lessons = [
            {"content": long_content, "cross_project": True, "project": "proj"},
        ]
        mock_welcome = {"greeting": "Hello", "memory_count": 10, "recent_memories": []}

        with patch("omega.bridge.welcome", return_value=mock_welcome), \
             patch("omega.bridge.consolidate"), \
             patch("omega.bridge.compact"), \
             patch("omega.bridge.get_cross_project_lessons", return_value=mock_lessons):
            session_start.main()

        output = capsys.readouterr().out
        assert "X" * 121 not in output

    def test_max_3_lessons(self, fake_home, tmp_omega_dir, capsys):
        import importlib
        import session_start
        importlib.reload(session_start)

        mock_lessons = [
            {"content": f"Lesson {i}", "cross_project": True, "project": f"p{i}"}
            for i in range(5)
        ]
        mock_welcome = {"greeting": "Hello", "memory_count": 10, "recent_memories": []}

        with patch("omega.bridge.welcome", return_value=mock_welcome), \
             patch("omega.bridge.consolidate"), \
             patch("omega.bridge.compact"), \
             patch("omega.bridge.get_cross_project_lessons", return_value=mock_lessons):
            session_start.main()

        output = capsys.readouterr().out
        # Count lines with project brackets (the lesson lines)
        lesson_lines = [line for line in output.splitlines() if line.strip().startswith("- [p")]
        assert len(lesson_lines) == 3

    def test_unknown_project_fallback(self, fake_home, tmp_omega_dir, capsys):
        import importlib
        import session_start
        importlib.reload(session_start)

        mock_lessons = [
            {"content": "Some lesson", "cross_project": True},  # no "project" key
        ]
        mock_welcome = {"greeting": "Hello", "memory_count": 10, "recent_memories": []}

        with patch("omega.bridge.welcome", return_value=mock_welcome), \
             patch("omega.bridge.consolidate"), \
             patch("omega.bridge.compact"), \
             patch("omega.bridge.get_cross_project_lessons", return_value=mock_lessons):
            session_start.main()

        output = capsys.readouterr().out
        assert "[unknown]" in output


# ============================================================================
# Hook server handle_session_start output format
# ============================================================================

class TestHookServerSessionStart:
    """Test handle_session_start output in daemon mode."""

    def test_header_with_memories(self):
        from omega.server.hook_server import handle_session_start
        mock_ctx = {
            "memory_count": 5, "health_status": "ok",
            "last_capture_ago": "5m ago", "context_items": [],
        }
        with patch("omega.bridge.get_session_context", return_value=mock_ctx), \
             patch("omega.bridge.consolidate"), \
             patch("omega.bridge.compact"):
            result = handle_session_start({"session_id": "s1", "project": "/p"})
        assert "## Welcome back! OMEGA ready" in result["output"]
        assert "5 memories" in result["output"]

    def test_context_items_in_output(self):
        from omega.server.hook_server import handle_session_start
        mock_ctx = {
            "memory_count": 42, "health_status": "ok",
            "last_capture_ago": "5m ago",
            "context_items": [
                {"tag": "DECISION", "text": "Use SQLite WAL mode"},
                {"tag": "LESSON", "text": "Lock is non-reentrant"},
            ],
        }
        with patch("omega.bridge.get_session_context", return_value=mock_ctx), \
             patch("omega.bridge.consolidate"), \
             patch("omega.bridge.compact"):
            result = handle_session_start({"session_id": "s1", "project": "/p"})
        assert "[CONTEXT]" in result["output"]
        assert "DECISION: Use SQLite WAL mode" in result["output"]
        assert "LESSON: Lock is non-reentrant" in result["output"]

    def test_first_session_greeting(self):
        from omega.server.hook_server import handle_session_start
        mock_ctx = {
            "memory_count": 0, "health_status": "ok",
            "last_capture_ago": "unknown", "context_items": [],
        }
        with patch("omega.bridge.get_session_context", return_value=mock_ctx), \
             patch("omega.bridge.consolidate"), \
             patch("omega.bridge.compact"):
            result = handle_session_start({"session_id": "s1", "project": "/p"})
        assert "## Welcome! OMEGA ready" in result["output"]
        assert "0 memories" in result["output"]

    def test_no_error_on_success(self):
        from omega.server.hook_server import handle_session_start
        mock_ctx = {
            "memory_count": 0, "health_status": "ok",
            "last_capture_ago": "unknown", "context_items": [],
        }
        with patch("omega.bridge.get_session_context", return_value=mock_ctx), \
             patch("omega.bridge.consolidate"), \
             patch("omega.bridge.compact"):
            result = handle_session_start({"session_id": "s1", "project": "/p"})
        assert result["error"] is None


# ============================================================================
# Stale surfacing file cleanup (both patterns)
# ============================================================================

class TestStaleSurfacingCleanup:
    """Test that session_start cleans up both .surfaced and .surfaced.json."""

    def test_cleans_old_surfaced_files(self, fake_home, tmp_omega_dir):
        import importlib
        import session_start
        importlib.reload(session_start)

        # Create stale files (old mtime)
        stale = tmp_omega_dir / "session-old.surfaced"
        stale.write_text("xxx")
        os.utime(stale, (0, 0))  # epoch = very old

        stale_json = tmp_omega_dir / "session-old.surfaced.json"
        stale_json.write_text("{}")
        os.utime(stale_json, (0, 0))

        # Create fresh files
        fresh = tmp_omega_dir / "session-new.surfaced"
        fresh.write_text("x")

        mock_welcome = {"greeting": "Hello", "memory_count": 0, "recent_memories": []}
        with patch("omega.bridge.welcome", return_value=mock_welcome), \
             patch("omega.bridge.consolidate"), \
             patch("omega.bridge.compact"), \
             patch("omega.bridge.get_cross_project_lessons", return_value=[]):
            session_start.main()

        assert not stale.exists()
        assert not stale_json.exists()
        assert fresh.exists()
