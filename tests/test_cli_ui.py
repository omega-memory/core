"""Tests for omega.cli_ui — Rich rendering helpers and plain-text fallback."""

import json
from io import StringIO
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# Rich-mode tests (capture via Rich Console → StringIO)
# ---------------------------------------------------------------------------


class TestRichMode:
    """Tests that run with Rich available (default)."""

    def test_print_header_rich(self):
        from omega.cli_ui import RICH_AVAILABLE

        if not RICH_AVAILABLE:
            pytest.skip("Rich not available")

        from rich.console import Console

        buf = StringIO()
        console = Console(file=buf, force_terminal=True, width=80)

        with patch("omega.cli_ui.console", console):
            from omega.cli_ui import print_header

            print_header("Test Header")

        output = buf.getvalue()
        assert "Test Header" in output

    def test_print_kv_rich(self):
        from omega.cli_ui import RICH_AVAILABLE

        if not RICH_AVAILABLE:
            pytest.skip("Rich not available")

        from rich.console import Console

        buf = StringIO()
        console = Console(file=buf, force_terminal=True, width=80)

        with patch("omega.cli_ui.console", console):
            from omega.cli_ui import print_kv

            print_kv([("Backend", "SQLite"), ("Size", "1.5 MB")])

        output = buf.getvalue()
        assert "Backend" in output
        assert "SQLite" in output
        assert "Size" in output

    def test_print_table_rich(self):
        from omega.cli_ui import RICH_AVAILABLE

        if not RICH_AVAILABLE:
            pytest.skip("Rich not available")

        from rich.console import Console

        buf = StringIO()
        console = Console(file=buf, force_terminal=True, width=80)

        with patch("omega.cli_ui.console", console):
            from omega.cli_ui import print_table

            print_table("My Table", ["Name", "Value"], [("alpha", "1"), ("beta", "2")])

        output = buf.getvalue()
        assert "alpha" in output
        assert "beta" in output

    def test_print_bar_chart_rich(self):
        from omega.cli_ui import RICH_AVAILABLE

        if not RICH_AVAILABLE:
            pytest.skip("Rich not available")

        from rich.console import Console

        buf = StringIO()
        console = Console(file=buf, force_terminal=True, width=80)

        with patch("omega.cli_ui.console", console):
            from omega.cli_ui import print_bar_chart

            print_bar_chart([("decision", 50), ("lesson", 30), ("error", 20)], title="Types")

        output = buf.getvalue()
        assert "decision" in output
        assert "50" in output

    def test_print_status_line_rich(self):
        from omega.cli_ui import RICH_AVAILABLE

        if not RICH_AVAILABLE:
            pytest.skip("Rich not available")

        from rich.console import Console

        buf = StringIO()
        console = Console(file=buf, force_terminal=True, width=80)

        with patch("omega.cli_ui.console", console):
            from omega.cli_ui import print_status_line

            print_status_line("ok", "All good")
            print_status_line("fail", "Something broke")
            print_status_line("warn", "Watch out")

        output = buf.getvalue()
        assert "All good" in output
        assert "Something broke" in output
        assert "Watch out" in output

    def test_print_summary_rich(self):
        from omega.cli_ui import RICH_AVAILABLE

        if not RICH_AVAILABLE:
            pytest.skip("Rich not available")

        from rich.console import Console

        buf = StringIO()
        console = Console(file=buf, force_terminal=True, width=80)

        with patch("omega.cli_ui.console", console):
            from omega.cli_ui import print_summary

            print_summary(0, 0)

        output = buf.getvalue()
        assert "passed" in output


# ---------------------------------------------------------------------------
# Plain-text fallback tests (NO_COLOR=1)
# ---------------------------------------------------------------------------


class TestPlainFallback:
    """Tests that simulate NO_COLOR=1 by patching RICH_AVAILABLE."""

    def test_print_header_plain(self, capsys):
        with patch("omega.cli_ui.RICH_AVAILABLE", False):
            from omega.cli_ui import print_header

            print_header("Test Title")

        out = capsys.readouterr().out
        assert "=== Test Title ===" in out

    def test_print_section_plain(self, capsys):
        with patch("omega.cli_ui.RICH_AVAILABLE", False):
            from omega.cli_ui import print_section

            print_section("My Section")

        out = capsys.readouterr().out
        assert "--- My Section ---" in out

    def test_print_kv_plain(self, capsys):
        with patch("omega.cli_ui.RICH_AVAILABLE", False):
            from omega.cli_ui import print_kv

            print_kv([("Key1", "Val1"), ("Key2", "Val2")])

        out = capsys.readouterr().out
        assert "Key1: Val1" in out
        assert "Key2: Val2" in out

    def test_print_table_plain(self, capsys):
        with patch("omega.cli_ui.RICH_AVAILABLE", False):
            from omega.cli_ui import print_table

            print_table("Title", ["A", "B"], [("x", "y"), ("m", "n")])

        out = capsys.readouterr().out
        assert "Title" in out
        assert "x" in out
        assert "n" in out

    def test_print_table_empty(self, capsys):
        with patch("omega.cli_ui.RICH_AVAILABLE", False):
            from omega.cli_ui import print_table

            print_table("Empty", ["Col"], [])

        out = capsys.readouterr().out
        assert "(empty)" in out

    def test_print_bar_chart_plain(self, capsys):
        with patch("omega.cli_ui.RICH_AVAILABLE", False):
            from omega.cli_ui import print_bar_chart

            print_bar_chart([("typeA", 80), ("typeB", 20)], title="Distribution")

        out = capsys.readouterr().out
        assert "typeA" in out
        assert "#" in out
        assert "80.0%" in out

    def test_print_bar_chart_zero_total(self, capsys):
        with patch("omega.cli_ui.RICH_AVAILABLE", False):
            from omega.cli_ui import print_bar_chart

            print_bar_chart([], title="Empty")

        out = capsys.readouterr().out
        assert "no data" in out

    def test_print_status_line_plain(self, capsys):
        with patch("omega.cli_ui.RICH_AVAILABLE", False):
            from omega.cli_ui import print_status_line

            print_status_line("ok", "Check passed")
            print_status_line("fail", "Check failed")
            print_status_line("warn", "Warning found")

        out = capsys.readouterr().out
        assert "[OK]" in out
        assert "[FAIL]" in out
        assert "[WARN]" in out

    def test_print_summary_plain(self, capsys):
        with patch("omega.cli_ui.RICH_AVAILABLE", False):
            from omega.cli_ui import print_summary

            print_summary(2, 3)

        out = capsys.readouterr().out
        assert "2 error(s)" in out
        assert "3 warning(s)" in out


# ---------------------------------------------------------------------------
# cmd_activity tests
# ---------------------------------------------------------------------------


class TestCmdActivity:
    """Tests for the omega activity command."""

    def _make_args(self, days=7, use_json=False):
        """Create a mock args namespace."""
        import argparse

        return argparse.Namespace(days=days, json=use_json)

    def test_activity_json(self, capsys):
        mock_data = {
            "sessions": [
                {
                    "session_id": "abc123",
                    "project": "/test",
                    "task": "testing",
                    "started_at": "2026-01-01",
                    "last_heartbeat": "",
                    "status": "active",
                }
            ],
            "tasks": [{"id": 1, "title": "Do thing", "status": "pending", "progress": 0, "created_at": "2026-01-01"}],
            "insights": [{"type": "decision", "preview": "Use Rich", "created_at": "2026-01-01", "id": "abc123abc123"}],
            "claims": [],
        }
        with patch("omega.bridge.get_activity_summary", return_value=mock_data):
            from omega.cli import cmd_activity

            cmd_activity(self._make_args(use_json=True))

        out = capsys.readouterr().out
        parsed = json.loads(out)
        assert len(parsed["sessions"]) == 1
        assert parsed["sessions"][0]["session_id"] == "abc123"
        assert len(parsed["tasks"]) == 1

    def test_activity_rich(self, capsys):
        mock_data = {
            "sessions": [
                {
                    "session_id": "sess-001",
                    "project": "/my/proj",
                    "task": "build",
                    "started_at": "2026-01-01T10:00:00",
                    "last_heartbeat": "",
                    "status": "active",
                }
            ],
            "tasks": [],
            "insights": [
                {
                    "type": "lesson_learned",
                    "preview": "Always test first",
                    "created_at": "2026-01-01T10:00:00",
                    "id": "mem123",
                }
            ],
            "claims": [{"type": "file", "path": "/foo/bar.py", "session": "sess-001"}],
        }
        with patch("omega.bridge.get_activity_summary", return_value=mock_data):
            from omega.cli import cmd_activity

            cmd_activity(self._make_args())

        # Just verify no crash — output format varies by Rich availability

    def test_activity_empty(self, capsys):
        mock_data = {"sessions": [], "tasks": [], "insights": [], "claims": []}
        with patch("omega.bridge.get_activity_summary", return_value=mock_data):
            from omega.cli import cmd_activity

            cmd_activity(self._make_args())

        out = capsys.readouterr().out
        assert "No active sessions" in out or "sessions" in out.lower() or "Session" in out
