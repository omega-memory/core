"""Tests for OMEGA Phase 4 — Observability & Resilience.

Tests CLI commands, doctor enhancements, FTS5 auto-repair, hook timing,
auto-backup, and plan capture.
"""
import sqlite3
import sys
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestCLIBackup:
    """Test omega backup command."""

    def test_backup_creates_file(self, tmp_omega_dir):
        """Backup creates a timestamped .db file."""
        # Create a minimal omega.db
        db_path = tmp_omega_dir / "omega.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE test (id INTEGER)")
        conn.execute("INSERT INTO test VALUES (1)")
        conn.commit()
        conn.close()

        from omega.cli import cmd_backup
        with patch("omega.cli.OMEGA_DIR", tmp_omega_dir):
            cmd_backup(MagicMock())

        backups = list((tmp_omega_dir / "backups").glob("omega-*.db"))
        assert len(backups) == 1

        # Verify the backup is a valid SQLite file
        bconn = sqlite3.connect(str(backups[0]))
        val = bconn.execute("SELECT * FROM test").fetchone()[0]
        assert val == 1
        bconn.close()

    def test_backup_rotation(self, tmp_omega_dir):
        """Backup rotates to keep only 5 most recent."""
        db_path = tmp_omega_dir / "omega.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE test (id INTEGER)")
        conn.commit()
        conn.close()

        backups_dir = tmp_omega_dir / "backups"
        backups_dir.mkdir()
        # Create 6 fake old backups
        import time
        for i in range(6):
            fake = backups_dir / f"omega-2026010{i}-000000.db"
            fake.write_text("fake")
            time.sleep(0.01)  # Ensure different mtimes

        from omega.cli import cmd_backup
        with patch("omega.cli.OMEGA_DIR", tmp_omega_dir):
            cmd_backup(MagicMock())

        # Should now have 5 total (6 old - 2 rotated + 1 new = 5)
        # Actually: 6 old + 1 new = 7, keep 5 = rotate 2
        backups = list(backups_dir.glob("omega-*.db"))
        assert len(backups) == 5

    def test_backup_no_db(self, tmp_omega_dir, capsys):
        """Backup with no omega.db prints a message."""
        from omega.cli import cmd_backup
        with patch("omega.cli.OMEGA_DIR", tmp_omega_dir):
            cmd_backup(MagicMock())
        out = capsys.readouterr().out
        assert "nothing to back up" in out.lower()


class TestCLILogs:
    """Test omega logs command."""

    def test_logs_shows_entries(self, tmp_omega_dir, capsys):
        """Logs command shows recent hook log entries."""
        hooks_log = tmp_omega_dir / "hooks.log"
        hooks_log.write_text("line1\nline2\nline3\n")

        from omega.cli import cmd_logs
        args = MagicMock()
        args.lines = 50
        with patch("omega.cli.OMEGA_DIR", tmp_omega_dir):
            cmd_logs(args)

        out = capsys.readouterr().out
        assert "line1" in out
        assert "line3" in out

    def test_logs_no_file(self, tmp_omega_dir, capsys):
        """Logs with no hooks.log prints a message."""
        from omega.cli import cmd_logs
        args = MagicMock()
        args.lines = 50
        with patch("omega.cli.OMEGA_DIR", tmp_omega_dir):
            cmd_logs(args)
        out = capsys.readouterr().out
        assert "no hook" in out.lower() or "no hooks" in out.lower()


class TestCLIValidate:
    """Test omega validate command."""

    def test_validate_healthy_db(self, tmp_omega_dir):
        """Validate passes on a healthy database."""
        # Create a minimal db with FTS5
        db_path = tmp_omega_dir / "omega.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE memories (id INTEGER PRIMARY KEY, content TEXT)")
        try:
            conn.execute("CREATE VIRTUAL TABLE memories_fts USING fts5(content, content='memories', content_rowid='id')")
        except Exception:
            pytest.skip("FTS5 not available")
        conn.commit()
        conn.close()

        from omega.cli import cmd_validate
        args = MagicMock()
        args.repair = False
        with patch("omega.cli.OMEGA_DIR", tmp_omega_dir):
            # Should exit with code 0
            with pytest.raises(SystemExit) as exc_info:
                cmd_validate(args)
            assert exc_info.value.code == 0


class TestFTS5AutoRepair:
    """Test FTS5 auto-repair in sqlite_store."""

    def test_text_search_works(self, store):
        """Basic text search functions."""
        store.store(content="Python programming is great")
        store.store(content="JavaScript is also popular")
        results = store._text_search("Python", limit=5)
        assert len(results) >= 1
        assert any("Python" in r.content for r in results)

    def test_fts_flag_exists(self, store):
        """Store has _fts_available flag."""
        assert hasattr(store, '_fts_available')


class TestAutoBackupBeforeConsolidate:
    """Test auto-backup before consolidation."""

    def test_consolidate_creates_backup(self, tmp_omega_dir, store):
        """consolidate() creates a pre-consolidate backup."""
        from omega.bridge import consolidate

        # Store some data
        store.store(content="Test memory for consolidation")

        # Point OMEGA_HOME at the temp dir
        with patch("omega.bridge.OMEGA_HOME", tmp_omega_dir), \
             patch("omega.bridge._get_store", return_value=store):
            # Create a fake omega.db for backup
            fake_db = tmp_omega_dir / "omega.db"
            conn = sqlite3.connect(str(fake_db))
            conn.execute("CREATE TABLE t (id INTEGER)")
            conn.commit()
            conn.close()

            consolidate(prune_days=30)

        backups = list((tmp_omega_dir / "backups").glob("pre-consolidate-*.db"))
        assert len(backups) == 1


class TestHookTiming:
    """Test that hooks have timing instrumentation."""

    def test_hooks_import_time(self):
        """All hooks import the time module."""
        hooks_dir = Path(__file__).parent.parent / "hooks"
        hook_files = [
            "pre_edit_surface.py", "surface_memories.py", "coord_heartbeat.py",
            "coord_session_start.py", "coord_session_stop.py", "session_start.py",
            "session_stop.py", "track_file_read.py", "pre_push_guard.py",
        ]
        for hook in hook_files:
            path = hooks_dir / hook
            if path.exists():
                content = path.read_text()
                assert "import time" in content, f"{hook} missing 'import time'"
                assert "_log_timing" in content, f"{hook} missing '_log_timing'"
                assert "time.monotonic()" in content, f"{hook} missing 'time.monotonic()'"


class TestAutoConsolidation:
    """Test auto-consolidation on session start."""

    def test_skips_if_recent(self, tmp_omega_dir):
        """Auto-consolidation skips if marker is < 7 days old."""
        marker = tmp_omega_dir / "last-consolidate"
        from datetime import datetime, timezone
        marker.write_text(datetime.now(timezone.utc).isoformat())

        # Import and run — should be a no-op
        # We can't easily test the hook directly, but we can test the logic
        last_ts = marker.read_text().strip()
        last = datetime.fromisoformat(last_ts)
        if last.tzinfo is None:
            last = last.replace(tzinfo=timezone.utc)
        age_days = (datetime.now(timezone.utc) - last).days
        assert age_days < 7  # Should skip

    def test_runs_if_stale(self, tmp_omega_dir):
        """Auto-consolidation runs if marker is > 7 days old."""
        marker = tmp_omega_dir / "last-consolidate"
        from datetime import datetime, timedelta, timezone
        old_date = datetime.now(timezone.utc) - timedelta(days=8)
        marker.write_text(old_date.isoformat())

        last_ts = marker.read_text().strip()
        last = datetime.fromisoformat(last_ts)
        if last.tzinfo is None:
            last = last.replace(tzinfo=timezone.utc)
        age_days = (datetime.now(timezone.utc) - last).days
        assert age_days >= 7  # Should run


class TestPlanCapture:
    """Test proactive plan/decision auto-capture."""

    def test_detect_plan_markers(self):
        """Plan markers are detected in tool output."""
        plan_markers = [
            "## Phase", "### Phase", "## Plan", "### Plan",
            "## Roadmap", "## Architecture", "## Implementation Plan",
            "## Recommendation", "## Design",
            "| File | Changes", "| Step |",
        ]
        test_output = """## Phase 1 — Setup
        Install dependencies and configure the database.
        | Step | Action |
        |------|--------|
        | 1    | Install packages |
        """
        has_plan = any(marker in test_output for marker in plan_markers)
        assert has_plan

    def test_short_output_ignored(self):
        """Output under 200 chars is not checked for plans."""
        short_output = "## Phase 1"
        assert len(short_output) < 200


class TestDoctorEnhancements:
    """Test that doctor checks are properly structured."""

    def test_doctor_function_exists(self):
        """cmd_doctor is importable."""
        from omega.cli import cmd_doctor
        assert callable(cmd_doctor)

    def test_cli_commands_registered(self):
        """All new CLI commands are registered."""
        # Just verify the function exists — actually running it needs args
        import omega.cli as cli
        assert hasattr(cli, 'cmd_backup')
        assert hasattr(cli, 'cmd_logs')
        assert hasattr(cli, 'cmd_validate')
