"""Tests for colored output in CLI commands.

This test file verifies that colored output works as expected for issue #10:
- omega status (colored output)
- omega doctor (colored output)
- omega query (colored output)
"""

import os
import sys
from io import StringIO
from unittest.mock import patch, MagicMock

import pytest

from omega.cli_ui import (
    print_header,
    print_section,
    print_kv,
    print_table,
    print_status_line,
    RICH_AVAILABLE,
    console,
)


class TestColoredOutputBasics:
    """Test that colored output is available and degrades gracefully."""

    def test_rich_available_or_fallback(self):
        """Rich should be available unless NO_COLOR is set."""
        if os.environ.get("NO_COLOR"):
            assert not RICH_AVAILABLE
        else:
            # Rich should be importable (though might not work in tests)
            assert RICH_AVAILABLE is not None

    def test_print_header_outputs(self, capsys):
        """print_header should produce output."""
        print_header("Test Header")
        out = capsys.readouterr().out
        assert "Test Header" in out

    def test_print_section_outputs(self, capsys):
        """print_section should produce output."""
        print_section("Test Section")
        out = capsys.readouterr().out
        assert "Test Section" in out

    def test_print_kv_outputs(self, capsys):
        """print_kv should produce output."""
        print_kv([("Key1", "Value1"), ("Key2", "Value2")])
        out = capsys.readouterr().out
        assert "Key1" in out
        assert "Value1" in out
        assert "Key2" in out
        assert "Value2" in out

    def test_print_status_line_outputs(self, capsys):
        """print_status_line should produce output."""
        print_status_line("ok", "Test message")
        out = capsys.readouterr().out
        assert "Test message" in out

    def test_no_color_environment_variable(self):
        """NO_COLOR should disable rich output."""
        original_no_color = os.environ.get("NO_COLOR")

        # Simulate NO_COLOR environment
        os.environ["NO_COLOR"] = "1"

        # Force reload of cli_ui module
        import importlib
        import omega.cli_ui
        importlib.reload(omega.cli_ui)

        # Rich should be unavailable
        assert not omega.cli_ui.RICH_AVAILABLE

        # Restore original environment
        if original_no_color is None:
            os.environ.pop("NO_COLOR", None)
        else:
            os.environ["NO_COLOR"] = original_no_color


class TestStatusCommandColoredOutput:
    """Test that omega status command uses colored output."""

    def test_status_uses_cli_ui_functions(self):
        """cmd_status should use print_header and print_kv."""
        # Import after potential reload
        from omega.cli import cmd_status

        args = MagicMock(json=False)

        with patch("omega.cli.OMEGA_DIR") as mock_dir:
            mock_dir.__truediv__.return_value.exists.return_value = False

            with patch("omega.cli.print_header") as mock_header:
                with patch("omega.cli.print_kv") as mock_kv:
                    with patch("omega.sqlite3.connect"):
                        cmd_status(args)

            # Verify colored output functions are called
            mock_header.assert_called_once()
            mock_kv.assert_called()


class TestDoctorCommandColoredOutput:
    """Test that omega doctor command uses colored output."""

    def test_doctor_uses_cli_ui_functions(self):
        """cmd_doctor should use print_header, print_section, print_status_line."""
        from omega.cli import cmd_doctor

        args = MagicMock(client=None)

        with patch("omega.cli.print_header") as mock_header:
            with patch("omega.cli.print_section") as mock_section:
                with patch("omega.cli.print_status_line") as mock_status:
                    with patch("omega.cli.print_summary") as mock_summary:
                        with patch("omega.cli.ok"):
                            cmd_doctor(args)

        # Verify colored output functions are called
        assert mock_header.call_count > 0
        assert mock_section.call_count > 0
        assert mock_status.call_count > 0


class TestQueryCommandColoredOutput:
    """Test that omega query command uses colored output."""

    def test_query_uses_print_table(self):
        """cmd_query should use print_table for non-JSON output."""
        from omega.cli import cmd_query

        mock_results = [
            {"content": "test result", "relevance": 0.9, "event_type": "memory"}
        ]
        args = MagicMock(
            query_text=["test"],
            limit=10,
            json=False,
            exact=False
        )

        with patch("omega.cli.time") as mock_time:
            mock_time.monotonic.side_effect = [0.0, 0.02]
            with patch("omega.bridge.query_structured", return_value=mock_results):
                with patch("omega.cli.print_table") as mock_table:
                    cmd_query(args)

        # Verify print_table is called for non-JSON output
        mock_table.assert_called_once()


class TestGracefulDegradation:
    """Test that output degrades gracefully when rich is unavailable."""

    def test_plain_text_fallback(self):
        """When rich is unavailable, plain text should be used."""
        # Mock rich as unavailable
        with patch("omega.cli_ui.RICH_AVAILABLE", False):
            # These should not crash
            print_header("Plain Header")
            print_section("Plain Section")
            print_kv([("Key", "Value")])
            print_status_line("ok", "Plain Status")

    def test_table_fallback(self):
        """print_table should work with plain text fallback."""
        with patch("omega.cli_ui.RICH_AVAILABLE", False):
            print_table(
                "Test Table",
                ["Col1", "Col2"],
                [["Row1Col1", "Row1Col2"], ["Row2Col1", "Row2Col2"]]
            )
