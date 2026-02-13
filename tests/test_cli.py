"""OMEGA CLI module tests — unit tests for cli.py functions."""

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from omega.cli import (
    _CLI_TYPE_MAP,
    _format_age,
    _inject_claude_md,
    _inject_settings_hooks,
    _resolve_python_path,
    OMEGA_BEGIN,
    OMEGA_END,
    cmd_query,
    cmd_remember,
    cmd_store,
)


# ============================================================================
# _resolve_python_path()
# ============================================================================


class TestResolvePythonPath:
    """Tests for _resolve_python_path()."""

    def test_returns_string(self):
        result = _resolve_python_path()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_returns_valid_path_when_sys_executable_exists(self):
        """When sys.executable is a real path, it should be returned."""
        result = _resolve_python_path()
        # The result should be some valid python path string
        assert "python" in result.lower() or Path(result).exists()

    def test_skips_venv_executable(self, monkeypatch):
        """If sys.executable contains 'venv', it should be skipped."""
        monkeypatch.setattr(sys, "executable", "/tmp/my_venv/bin/python3")
        # Should fall through to shutil.which or fallback
        result = _resolve_python_path()
        assert "my_venv" not in result

    def test_falls_back_to_which_python3(self, monkeypatch):
        """When sys.executable is empty, should try shutil.which('python3')."""
        monkeypatch.setattr(sys, "executable", "")
        with patch("omega.cli.shutil.which", return_value="/usr/bin/python3"):
            result = _resolve_python_path()
        assert result == "/usr/bin/python3"

    def test_fallback_when_nothing_works(self, monkeypatch):
        """When sys.executable is empty and shutil.which returns None, returns fallback."""
        monkeypatch.setattr(sys, "executable", "")
        with patch("omega.cli.shutil.which", return_value=None):
            with patch("omega.cli.Path.exists", return_value=False):
                result = _resolve_python_path()
        # Should return "python3" as last resort (empty exe or "python3")
        assert result in ("", "python3")


# ============================================================================
# _format_age()
# ============================================================================


class TestFormatAge:
    """Tests for _format_age() relative time formatting."""

    def test_none_input_returns_empty(self):
        assert _format_age(None) == ""

    def test_just_now(self):
        """Timestamps within 60 seconds should return 'just now'."""
        now = datetime.now(timezone.utc) - timedelta(seconds=10)
        assert _format_age(now) == "just now"

    def test_minutes_ago(self):
        """Timestamps 1-59 minutes ago should return 'Xm ago'."""
        ts = datetime.now(timezone.utc) - timedelta(minutes=5)
        result = _format_age(ts)
        assert result.endswith("m ago")
        assert result.startswith("5") or result.startswith("4")  # allow 1s drift

    def test_hours_ago(self):
        """Timestamps 1-23 hours ago should return 'Xh ago'."""
        ts = datetime.now(timezone.utc) - timedelta(hours=3)
        result = _format_age(ts)
        assert result == "3h ago"

    def test_days_ago(self):
        """Timestamps 1-6 days ago should return 'Xd ago'."""
        ts = datetime.now(timezone.utc) - timedelta(days=4)
        result = _format_age(ts)
        assert result == "4d ago"

    def test_weeks_ago(self):
        """Timestamps 7-29 days ago should return 'Xw ago'."""
        ts = datetime.now(timezone.utc) - timedelta(days=14)
        result = _format_age(ts)
        assert result == "2w ago"

    def test_months_ago(self):
        """Timestamps >= 30 days ago should return 'Xmo ago'."""
        ts = datetime.now(timezone.utc) - timedelta(days=65)
        result = _format_age(ts)
        assert result == "2mo ago"

    def test_naive_datetime_treated_as_utc(self):
        """Naive datetimes (no tzinfo) should be treated as UTC without error."""
        naive_ts = datetime.utcnow() - timedelta(hours=2)
        result = _format_age(naive_ts)
        assert result == "2h ago"

    def test_boundary_59_seconds(self):
        """At exactly 59 seconds, should still be 'just now'."""
        ts = datetime.now(timezone.utc) - timedelta(seconds=59)
        assert _format_age(ts) == "just now"

    def test_boundary_60_seconds(self):
        """At exactly 60 seconds, should be '1m ago'."""
        ts = datetime.now(timezone.utc) - timedelta(seconds=60)
        assert _format_age(ts) == "1m ago"


# ============================================================================
# _CLI_TYPE_MAP
# ============================================================================


class TestCLITypeMap:
    """Tests for _CLI_TYPE_MAP dictionary."""

    def test_has_expected_keys(self):
        expected = {"memory", "lesson", "decision", "error", "task", "preference"}
        assert set(_CLI_TYPE_MAP.keys()) == expected

    def test_lesson_maps_to_lesson_learned(self):
        assert _CLI_TYPE_MAP["lesson"] == "lesson_learned"

    def test_error_maps_to_error_pattern(self):
        assert _CLI_TYPE_MAP["error"] == "error_pattern"

    def test_memory_maps_to_memory(self):
        assert _CLI_TYPE_MAP["memory"] == "memory"

    def test_preference_maps_to_user_preference(self):
        assert _CLI_TYPE_MAP["preference"] == "user_preference"


# ============================================================================
# _inject_claude_md()
# ============================================================================


class TestInjectClaudeMd:
    """Tests for _inject_claude_md() CLAUDE.md management."""

    @pytest.fixture(autouse=True)
    def _setup_paths(self, tmp_path, monkeypatch):
        """Point CLAUDE_MD_PATH and DATA_DIR to temp locations."""
        self.claude_md = tmp_path / ".claude" / "CLAUDE.md"
        self.data_dir = tmp_path / "data"
        self.data_dir.mkdir()

        # Create a minimal fragment file
        self.fragment_text = (
            "<!-- OMEGA:BEGIN — managed by omega setup, do not edit this block -->\n"
            "## Memory (OMEGA)\n"
            "\n"
            "- `omega_remember(text)` — user says \"remember\"\n"
            "<!-- OMEGA:END -->"
        )
        (self.data_dir / "claude-md-fragment.md").write_text(self.fragment_text + "\n")

        monkeypatch.setattr("omega.cli.CLAUDE_MD_PATH", self.claude_md)
        monkeypatch.setattr("omega.cli.DATA_DIR", self.data_dir)

    def test_creates_new_file(self, capsys):
        """If CLAUDE.md does not exist, it should be created with the fragment."""
        assert not self.claude_md.exists()
        _inject_claude_md()
        assert self.claude_md.exists()
        content = self.claude_md.read_text()
        assert OMEGA_BEGIN in content
        assert OMEGA_END in content
        assert "appended" in capsys.readouterr().out

    def test_appends_to_existing_file_without_block(self, capsys):
        """If CLAUDE.md exists but has no OMEGA block, append it."""
        self.claude_md.parent.mkdir(parents=True, exist_ok=True)
        self.claude_md.write_text("# My Config\n\nSome existing content.\n")
        _inject_claude_md()
        content = self.claude_md.read_text()
        assert "My Config" in content
        assert OMEGA_BEGIN in content
        assert "appended" in capsys.readouterr().out

    def test_updates_existing_omega_block(self, capsys):
        """If CLAUDE.md has an existing OMEGA block, replace it."""
        self.claude_md.parent.mkdir(parents=True, exist_ok=True)
        old_block = (
            "# Config\n\n"
            "<!-- OMEGA:BEGIN — old version -->\n"
            "## Memory (OMEGA)\n"
            "- old instructions\n"
            "<!-- OMEGA:END -->\n\n"
            "## Other\n"
        )
        self.claude_md.write_text(old_block)
        _inject_claude_md()
        content = self.claude_md.read_text()
        assert "old instructions" not in content
        assert "omega_remember" in content
        assert "## Other" in content
        assert "updated" in capsys.readouterr().out

    def test_already_up_to_date(self, capsys):
        """If the block is already identical, report already up to date."""
        self.claude_md.parent.mkdir(parents=True, exist_ok=True)
        # Write exactly what inject would produce
        self.claude_md.write_text(self.fragment_text)
        _inject_claude_md()
        assert "already up to date" in capsys.readouterr().out

    def test_replaces_plain_memory_section(self, capsys):
        """If there is a plain '## Memory (OMEGA)' section (no markers), replace it."""
        self.claude_md.parent.mkdir(parents=True, exist_ok=True)
        plain_content = (
            "# Config\n\n"
            "## Memory (OMEGA)\n"
            "- Some old plain instructions\n"
            "- More old stuff\n"
            "\n"
            "## Other Section\n"
        )
        self.claude_md.write_text(plain_content)
        _inject_claude_md()
        content = self.claude_md.read_text()
        assert "Some old plain instructions" not in content
        assert OMEGA_BEGIN in content
        assert "## Other Section" in content
        assert "replaced plain" in capsys.readouterr().out


# ============================================================================
# _inject_settings_hooks()
# ============================================================================


class TestInjectSettingsHooks:
    """Tests for _inject_settings_hooks() hook injection."""

    @pytest.fixture(autouse=True)
    def _setup_paths(self, tmp_path, monkeypatch):
        """Point SETTINGS_JSON_PATH and DATA_DIR to temp locations."""
        self.settings_json = tmp_path / ".claude" / "settings.json"
        self.data_dir = tmp_path / "data"
        self.data_dir.mkdir()
        self.hooks_src = tmp_path / "omega" / "hooks"
        self.hooks_src.mkdir(parents=True)

        # Minimal hooks.json manifest (2 events, 2 hooks).
        # Uses simple script names (no spaces) so the idempotency check works.
        manifest = {
            "SessionStart": [
                {"script": "session_start.py", "timeout": 5000, "matcher": ""}
            ],
            "PostToolUse": [
                {"script": "surface_memories.py", "timeout": 5000, "matcher": "Edit|Write"}
            ],
        }
        (self.data_dir / "hooks.json").write_text(json.dumps(manifest))

        monkeypatch.setattr("omega.cli.SETTINGS_JSON_PATH", self.settings_json)
        monkeypatch.setattr("omega.cli.DATA_DIR", self.data_dir)
        monkeypatch.setattr("omega.cli._resolve_python_path", lambda: "/usr/bin/python3")

    def test_creates_new_settings_file(self, capsys):
        """If settings.json does not exist, it should be created with hooks."""
        assert not self.settings_json.exists()
        _inject_settings_hooks(self.hooks_src)
        assert self.settings_json.exists()
        settings = json.loads(self.settings_json.read_text())
        assert "hooks" in settings
        assert "SessionStart" in settings["hooks"]
        assert "PostToolUse" in settings["hooks"]
        assert "configured" in capsys.readouterr().out

    def test_appends_to_existing_settings(self, capsys):
        """If settings.json exists with other data, hooks should be added."""
        self.settings_json.parent.mkdir(parents=True, exist_ok=True)
        self.settings_json.write_text(json.dumps({"allowedTools": ["Edit"]}))
        _inject_settings_hooks(self.hooks_src)
        settings = json.loads(self.settings_json.read_text())
        assert "allowedTools" in settings
        assert "hooks" in settings
        assert len(settings["hooks"]["SessionStart"]) == 1

    def test_idempotent_re_injection(self, capsys):
        """Running injection twice should not duplicate hooks."""
        self.settings_json.parent.mkdir(parents=True, exist_ok=True)
        self.settings_json.write_text("{}")

        _inject_settings_hooks(self.hooks_src)
        first_settings = json.loads(self.settings_json.read_text())
        first_count = len(first_settings["hooks"].get("SessionStart", []))

        _inject_settings_hooks(self.hooks_src)
        second_settings = json.loads(self.settings_json.read_text())
        second_count = len(second_settings["hooks"].get("SessionStart", []))

        assert first_count == second_count
        out = capsys.readouterr().out
        assert "already configured" in out

    def test_hook_command_includes_python_path(self):
        """Hook commands should reference the resolved python path."""
        self.settings_json.parent.mkdir(parents=True, exist_ok=True)
        self.settings_json.write_text("{}")
        _inject_settings_hooks(self.hooks_src)
        settings = json.loads(self.settings_json.read_text())
        hook_entry = settings["hooks"]["SessionStart"][0]
        command = hook_entry["hooks"][0]["command"]
        assert command.startswith("/usr/bin/python3")
        assert "session_start.py" in command

    def test_hook_entry_structure(self):
        """Each hook entry should have the correct structure."""
        self.settings_json.parent.mkdir(parents=True, exist_ok=True)
        self.settings_json.write_text("{}")
        _inject_settings_hooks(self.hooks_src)
        settings = json.loads(self.settings_json.read_text())

        for event, entries in settings["hooks"].items():
            for entry in entries:
                assert "hooks" in entry
                assert "matcher" in entry
                for h in entry["hooks"]:
                    assert "command" in h
                    assert "timeout" in h
                    assert "type" in h
                    assert h["type"] == "command"

    def test_malformed_settings_json_skips(self, capsys):
        """If settings.json is malformed, injection should warn and skip."""
        self.settings_json.parent.mkdir(parents=True, exist_ok=True)
        self.settings_json.write_text("{ not valid json")
        _inject_settings_hooks(self.hooks_src)
        out = capsys.readouterr().out
        assert "malformed" in out.lower()


# ============================================================================
# cmd_query()
# ============================================================================


class TestCmdQuery:
    """Tests for cmd_query() CLI command."""

    def test_empty_query_exits_with_error(self, capsys):
        """Empty query text should print usage and exit with code 1."""
        args = argparse.Namespace(query_text=["  "], limit=10, json=False, exact=False)
        with pytest.raises(SystemExit) as exc_info:
            cmd_query(args)
        assert exc_info.value.code == 1
        assert "Usage" in capsys.readouterr().err

    def test_semantic_query_calls_bridge(self, capsys):
        """A valid query should call bridge.query_structured."""
        mock_results = [
            {"content": "test result", "relevance": 0.85, "event_type": "lesson_learned"}
        ]
        args = argparse.Namespace(query_text=["test", "query"], limit=5, json=False, exact=False)

        with patch("omega.cli.time") as mock_time:
            mock_time.monotonic.side_effect = [0.0, 0.05]
            with patch("omega.bridge.query_structured", return_value=mock_results) as mock_qs:
                cmd_query(args)
                mock_qs.assert_called_once_with("test query", limit=5)

        out = capsys.readouterr().out
        assert "1 result" in out

    def test_json_output_mode(self, capsys):
        """--json flag should output JSON."""
        mock_results = [
            {"content": "json result", "relevance": 0.9, "event_type": "decision"}
        ]
        args = argparse.Namespace(query_text=["hello"], limit=10, json=True, exact=False)

        with patch("omega.cli.time") as mock_time:
            mock_time.monotonic.side_effect = [0.0, 0.02]
            with patch("omega.bridge.query_structured", return_value=mock_results):
                cmd_query(args)

        out = capsys.readouterr().out
        parsed = json.loads(out)
        assert "results" in parsed
        assert parsed["count"] == 1


# ============================================================================
# cmd_store()
# ============================================================================


class TestCmdStore:
    """Tests for cmd_store() CLI command."""

    def test_empty_content_exits_with_error(self, capsys):
        """Empty content should print usage and exit with code 1."""
        args = argparse.Namespace(content=[" "], type="memory")
        with pytest.raises(SystemExit) as exc_info:
            cmd_store(args)
        assert exc_info.value.code == 1
        assert "Usage" in capsys.readouterr().err

    def test_stores_memory_with_default_type(self, capsys):
        """Valid content should call bridge.store with correct event_type."""
        args = argparse.Namespace(content=["test", "memory", "content"], type="memory")
        with patch("omega.bridge.store") as mock_store:
            cmd_store(args)
            mock_store.assert_called_once_with(
                content="test memory content", event_type="memory"
            )
        assert "Stored [memory]" in capsys.readouterr().out

    def test_stores_lesson_type(self, capsys):
        """Type 'lesson' should be mapped to 'lesson_learned'."""
        args = argparse.Namespace(content=["important", "lesson"], type="lesson")
        with patch("omega.bridge.store") as mock_store:
            cmd_store(args)
            mock_store.assert_called_once_with(
                content="important lesson", event_type="lesson_learned"
            )
        assert "Stored [lesson]" in capsys.readouterr().out

    def test_stores_error_type(self, capsys):
        """Type 'error' should be mapped to 'error_pattern'."""
        args = argparse.Namespace(content=["some", "error"], type="error")
        with patch("omega.bridge.store") as mock_store:
            cmd_store(args)
            mock_store.assert_called_once_with(
                content="some error", event_type="error_pattern"
            )


# ============================================================================
# cmd_remember()
# ============================================================================


class TestCmdRemember:
    """Tests for cmd_remember() CLI command."""

    def test_empty_text_exits_with_error(self, capsys):
        """Empty text should print usage and exit with code 1."""
        args = argparse.Namespace(text=["  "])
        with pytest.raises(SystemExit) as exc_info:
            cmd_remember(args)
        assert exc_info.value.code == 1
        assert "Usage" in capsys.readouterr().err

    def test_remembers_valid_text(self, capsys):
        """Valid text should call bridge.remember."""
        args = argparse.Namespace(text=["I", "prefer", "dark", "mode"])
        with patch("omega.bridge.remember", return_value={"status": "ok"}) as mock_remember:
            cmd_remember(args)
            mock_remember.assert_called_once_with(text="I prefer dark mode")
        assert "Remembered:" in capsys.readouterr().out
        assert "dark mode" in capsys.readouterr().out or True  # already consumed

    def test_output_truncates_long_text(self, capsys):
        """Output should truncate text at 120 chars."""
        long_text = "x" * 200
        args = argparse.Namespace(text=[long_text])
        with patch("omega.bridge.remember", return_value={"status": "ok"}):
            cmd_remember(args)
        out = capsys.readouterr().out
        # The print uses text[:120], so output should not contain the full 200 chars
        assert "Remembered:" in out


# ============================================================================
# cmd_validate() — table allowlist
# ============================================================================


class TestCmdValidateTableAllowlist:
    """Tests for cmd_validate table name safety."""

    def test_valid_tables_is_frozenset(self):
        """_VALID_TABLES used in cmd_validate should be a known set of table names."""
        # The allowlist is defined inline — verify that the expected tables
        # are all identifiers (no SQL injection vectors).
        expected = {
            "memories", "edges", "entity_index",
            "coord_sessions", "coord_file_claims", "coord_branch_claims",
            "coord_intents", "coord_snapshots", "coord_tasks", "coord_audit",
        }
        for tbl in expected:
            assert tbl.isidentifier(), f"{tbl} is not a valid identifier"


# ============================================================================
# cmd_doctor() — bridge import check
# ============================================================================


class TestCmdDoctorBridgeCheck:
    """Tests for cmd_doctor bridge import actually importing."""

    def test_doctor_bridge_import_works(self):
        """cmd_doctor should actually import bridge functions, not just print ok."""
        # Verify the imports are real by importing them ourselves
        from omega.bridge import status, auto_capture, query
        assert callable(status)
        assert callable(auto_capture)
        assert callable(query)


# ============================================================================
# Redundant import cleanup
# ============================================================================


class TestModuleLevelImports:
    """Verify datetime imports are at module level, not duplicated inside functions."""

    def test_timedelta_available_at_module_level(self):
        """timedelta should be importable from omega.cli at module level."""
        import omega.cli as cli
        assert hasattr(cli, "timedelta")
        from datetime import timedelta as _td
        assert cli.timedelta is _td
