"""Tests for omega.migrate_to_sqlite — helper functions and auto-migration."""
import sys
from pathlib import Path
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from omega import json_compat as json
from omega.migrate_to_sqlite import (
    _load_jsonl_entries,
    _normalize_metadata,
    auto_migrate_if_needed,
)


# ============================================================================
# _normalize_metadata
# ============================================================================


class TestNormalizeMetadata:
    def test_empty_metadata(self):
        result = _normalize_metadata({})
        assert result == {}

    def test_none_metadata(self):
        result = _normalize_metadata({"metadata": None})
        assert result == {}

    def test_preserves_existing_event_type(self):
        node = {"metadata": {"event_type": "lesson_learned", "type": "old_type"}}
        result = _normalize_metadata(node)
        assert result["event_type"] == "lesson_learned"
        # "type" should still be present (not removed)
        assert result["type"] == "old_type"

    def test_promotes_type_to_event_type(self):
        node = {"metadata": {"type": "decision"}}
        result = _normalize_metadata(node)
        assert result["event_type"] == "decision"
        assert result["type"] == "decision"

    def test_returns_copy(self):
        """Should not mutate the original node metadata."""
        original_meta = {"key": "value"}
        node = {"metadata": original_meta}
        result = _normalize_metadata(node)
        result["new_key"] = "new_value"
        assert "new_key" not in original_meta


# ============================================================================
# _load_jsonl_entries
# ============================================================================


class TestLoadJsonlEntries:
    def test_nonexistent_file(self, tmp_path):
        result = _load_jsonl_entries(tmp_path / "missing.jsonl")
        assert result == []

    def test_empty_file(self, tmp_path):
        path = tmp_path / "empty.jsonl"
        path.write_text("")
        result = _load_jsonl_entries(path)
        assert result == []

    def test_valid_entries(self, tmp_path):
        path = tmp_path / "store.jsonl"
        entries = [
            {"content": "first", "event_type": "lesson_learned"},
            {"content": "second", "event_type": "decision"},
        ]
        path.write_text("\n".join(json.dumps(e) for e in entries))
        result = _load_jsonl_entries(path)
        assert len(result) == 2
        assert result[0]["content"] == "first"
        assert result[1]["content"] == "second"

    def test_skips_blank_lines(self, tmp_path):
        path = tmp_path / "store.jsonl"
        path.write_text('{"content": "a"}\n\n\n{"content": "b"}\n')
        result = _load_jsonl_entries(path)
        assert len(result) == 2

    def test_skips_malformed_lines(self, tmp_path):
        path = tmp_path / "store.jsonl"
        path.write_text('{"content": "good"}\nnot json\n{"content": "also good"}\n')
        result = _load_jsonl_entries(path)
        assert len(result) == 2
        assert result[0]["content"] == "good"
        assert result[1]["content"] == "also good"


# ============================================================================
# auto_migrate_if_needed
# ============================================================================


class TestAutoMigrateIfNeeded:
    def test_fresh_install_no_migration(self, tmp_omega_dir):
        """No graphs, no JSONL, no DB — returns False."""
        with patch("omega.migrate_to_sqlite.OMEGA_DIR", tmp_omega_dir), \
             patch("omega.migrate_to_sqlite.GRAPHS_DIR", tmp_omega_dir / "graphs"), \
             patch("omega.migrate_to_sqlite.DB_PATH", tmp_omega_dir / "omega.db"):
            assert auto_migrate_if_needed() is False

    def test_existing_db_with_data_skips(self, tmp_omega_dir):
        """If DB already has data, returns False."""
        import sqlite3
        db_path = tmp_omega_dir / "omega.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE memories (id TEXT, content TEXT)")
        conn.execute("INSERT INTO memories VALUES ('1', 'test')")
        conn.commit()
        conn.close()

        with patch("omega.migrate_to_sqlite.OMEGA_DIR", tmp_omega_dir), \
             patch("omega.migrate_to_sqlite.GRAPHS_DIR", tmp_omega_dir / "graphs"), \
             patch("omega.migrate_to_sqlite.DB_PATH", db_path):
            assert auto_migrate_if_needed() is False

    def test_existing_db_no_table_proceeds(self, tmp_omega_dir):
        """If DB exists but no memories table, should not crash."""
        import sqlite3
        db_path = tmp_omega_dir / "omega.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE other (id TEXT)")
        conn.commit()
        conn.close()

        # No graphs or JSONL either — should return False (nothing to migrate)
        with patch("omega.migrate_to_sqlite.OMEGA_DIR", tmp_omega_dir), \
             patch("omega.migrate_to_sqlite.GRAPHS_DIR", tmp_omega_dir / "graphs"), \
             patch("omega.migrate_to_sqlite.DB_PATH", db_path):
            assert auto_migrate_if_needed() is False

    def test_conn_closed_on_exception(self, tmp_omega_dir):
        """Connection should be closed even if execute raises."""
        import sqlite3
        db_path = tmp_omega_dir / "omega.db"
        # Create an empty DB (no tables)
        conn = sqlite3.connect(str(db_path))
        conn.close()

        with patch("omega.migrate_to_sqlite.OMEGA_DIR", tmp_omega_dir), \
             patch("omega.migrate_to_sqlite.GRAPHS_DIR", tmp_omega_dir / "graphs"), \
             patch("omega.migrate_to_sqlite.DB_PATH", db_path):
            # Should not raise — the try/finally should handle the missing table
            result = auto_migrate_if_needed()
            assert result is False

    def test_migration_failure_returns_false(self, tmp_omega_dir):
        """If migrate() raises, auto_migrate returns False."""
        graphs_dir = tmp_omega_dir / "graphs"
        graphs_dir.mkdir()
        # Create a minimal semantic.json to trigger migration
        (graphs_dir / "semantic.json").write_text('{"nodes": [{"id": "1", "content": "test"}]}')

        with patch("omega.migrate_to_sqlite.OMEGA_DIR", tmp_omega_dir), \
             patch("omega.migrate_to_sqlite.GRAPHS_DIR", graphs_dir), \
             patch("omega.migrate_to_sqlite.DB_PATH", tmp_omega_dir / "omega.db"), \
             patch("omega.migrate_to_sqlite.migrate", side_effect=RuntimeError("boom")):
            result = auto_migrate_if_needed()
            assert result is False
