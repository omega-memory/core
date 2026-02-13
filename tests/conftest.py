"""OMEGA test configuration."""
import os
import sys
import pytest
from pathlib import Path

# Ensure omega package and hooks are importable
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "hooks"))


@pytest.fixture
def tmp_omega_dir(tmp_path):
    """Create a temporary OMEGA directory for testing."""
    omega_dir = tmp_path / ".omega"
    omega_dir.mkdir()
    os.environ["OMEGA_HOME"] = str(omega_dir)
    yield omega_dir
    os.environ.pop("OMEGA_HOME", None)


@pytest.fixture(autouse=True)
def _reset_embeddings_after_test():
    """Reset embedding circuit-breaker after every test to prevent state leaks."""
    yield
    from omega.graphs import reset_embedding_state
    reset_embedding_state()


@pytest.fixture(autouse=True)
def _reset_hook_server_state():
    """Clear hook_server debounce dicts and entity engine cache before each test."""
    try:
        from omega.server import hook_server
        hook_server._last_claim.clear()
        hook_server._last_overlap_notify.clear()
        hook_server._last_coord_query.clear()
        hook_server._last_reminder_check = 0.0
        hook_server._pending_urgent.clear()
        hook_server._session_intent.clear()
    except (ImportError, AttributeError):
        pass
    try:
        import omega.entity.engine as ee
        ee._cache_ts = 0.0
    except (ImportError, AttributeError):
        pass
    yield
    try:
        from omega.server import hook_server
        hook_server._last_claim.clear()
        hook_server._last_overlap_notify.clear()
        hook_server._last_coord_query.clear()
        hook_server._last_reminder_check = 0.0
        hook_server._pending_urgent.clear()
        hook_server._session_intent.clear()
    except (ImportError, AttributeError):
        pass
    try:
        import omega.entity.engine as ee
        ee._cache_ts = 0.0
    except (ImportError, AttributeError):
        pass


@pytest.fixture
def store(tmp_omega_dir):
    """Create a fresh SQLiteStore for testing."""
    from omega.sqlite_store import SQLiteStore
    db_path = tmp_omega_dir / "test.db"
    s = SQLiteStore(db_path=db_path)
    yield s
    s.close()


@pytest.fixture
def coord_mgr(tmp_omega_dir):
    """Create a fresh CoordinationManager for testing."""
    from omega.coordination import CoordinationManager
    db_path = tmp_omega_dir / "test.db"
    mgr = CoordinationManager(db_path=db_path)
    yield mgr
    mgr.close()
