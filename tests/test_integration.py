"""OMEGA integration tests — verify the full stack works end-to-end."""

import os
import sys
import subprocess
from contextlib import contextmanager
import pytest
from pathlib import Path

# Ensure omega package is importable
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestPackageImport:
    """P0: The package must import without error."""

    def test_import_omega(self):
        from omega import __version__
        assert __version__

    def test_import_bridge(self):
        from omega.bridge import auto_capture, query, status
        assert callable(auto_capture)
        assert callable(query)
        assert callable(status)

    def test_import_handlers(self):
        from omega.server.handlers import HANDLERS
        assert len(HANDLERS) >= 20

    def test_import_tool_schemas(self):
        from omega.server.tool_schemas import TOOL_SCHEMAS
        assert len(TOOL_SCHEMAS) >= 20

    def test_import_types(self):
        from omega.types import AutoCaptureEventType, TTLCategory
        assert AutoCaptureEventType.GIT_CONFLICT == "git_conflict"
        assert TTLCategory.PERMANENT is None


@contextmanager
def _skip_embeddings():
    """Context manager that skips embeddings and resets the circuit breaker after."""
    from omega.graphs import reset_embedding_state
    os.environ["OMEGA_SKIP_EMBEDDINGS"] = "1"
    try:
        yield
    finally:
        os.environ.pop("OMEGA_SKIP_EMBEDDINGS", None)
        reset_embedding_state()


class TestDatabaseRoundtrip:
    """Verify store → query → delete works with a temp database."""

    def test_store_and_query(self, tmp_omega_dir):
        with _skip_embeddings():
            from omega.sqlite_store import SQLiteStore
            store = SQLiteStore(db_path=tmp_omega_dir / "test.db")

            # Store
            node_id = store.store(
                content="Test memory: Python prefers spaces over tabs",
                metadata={"event_type": "lesson_learned"},
            )
            assert node_id.startswith("mem-")

            # Query
            results = store.query("Python spaces tabs", limit=5)
            assert len(results) >= 1
            assert any("spaces" in r.content for r in results)

            # Delete
            deleted = store.delete_node(node_id)
            assert deleted is True

            # Verify gone
            assert store.get_node(node_id) is None

            store.close()

    def test_content_dedup(self, tmp_omega_dir):
        """Storing identical content twice should return the same node_id."""
        with _skip_embeddings():
            from omega.sqlite_store import SQLiteStore
            store = SQLiteStore(db_path=tmp_omega_dir / "test.db")

            id1 = store.store(content="Exact duplicate test content here")
            id2 = store.store(content="Exact duplicate test content here")
            assert id1 == id2
            assert store.node_count() == 1

            store.close()

    def test_null_content_rejected(self, tmp_omega_dir):
        """Storing empty content should raise ValueError."""
        with _skip_embeddings():
            from omega.sqlite_store import SQLiteStore
            store = SQLiteStore(db_path=tmp_omega_dir / "test.db")

            with pytest.raises(ValueError):
                store.store(content="")

            store.close()


class TestBridge:
    """Verify bridge-level API works."""

    def test_auto_capture_and_query(self, tmp_omega_dir):
        with _skip_embeddings():
            from omega.bridge import auto_capture, query, reset_memory
            reset_memory()

            result = auto_capture(
                content="Integration test: always use type hints in Python",
                event_type="lesson_learned",
                session_id="test-session",
            )
            assert "Memory Captured" in result or "Memory Deduplicated" in result

            query_result = query(query_text="type hints Python", limit=5)
            assert "type hints" in query_result

            reset_memory()


class TestHandlerValidation:
    """Verify handler input validation."""

    @pytest.mark.asyncio
    async def test_backup_export_rejects_path_traversal(self):
        from omega.server.handlers import handle_omega_backup
        result = await handle_omega_backup({"filepath": "/etc/passwd", "mode": "export"})
        assert result.get("isError")

    @pytest.mark.asyncio
    async def test_backup_import_rejects_path_traversal(self):
        from omega.server.handlers import handle_omega_backup
        result = await handle_omega_backup({"filepath": "/etc/passwd", "mode": "import"})
        assert result.get("isError")


class TestCLIDoctor:
    """Verify omega doctor runs without crashing."""

    def test_doctor_runs(self):
        result = subprocess.run(
            [sys.executable, "-m", "omega.cli", "doctor"],
            capture_output=True, text=True, timeout=30,
        )
        # Doctor may exit 0 or 1 depending on environment, but should not crash
        assert result.returncode in (0, 1)
        assert "OMEGA Doctor" in result.stdout
