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
    # Default: disable encryption in tests for deterministic output
    old_encrypt = os.environ.get("OMEGA_ENCRYPT")
    os.environ["OMEGA_ENCRYPT"] = "0"
    yield omega_dir
    os.environ.pop("OMEGA_HOME", None)
    if old_encrypt is not None:
        os.environ["OMEGA_ENCRYPT"] = old_encrypt
    else:
        os.environ.pop("OMEGA_ENCRYPT", None)
    from omega.crypto import reset_crypto_state
    reset_crypto_state()


@pytest.fixture
def tmp_omega_dir_encrypted(tmp_path):
    """Create a temporary OMEGA directory with encryption enabled."""
    omega_dir = tmp_path / ".omega"
    omega_dir.mkdir()
    os.environ["OMEGA_HOME"] = str(omega_dir)
    os.environ["OMEGA_ENCRYPT"] = "1"
    from omega.crypto import reset_crypto_state
    reset_crypto_state()
    yield omega_dir
    os.environ.pop("OMEGA_HOME", None)
    os.environ.pop("OMEGA_ENCRYPT", None)
    reset_crypto_state()


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
        hook_server._debounce_state.reset()
        hook_server._last_reminder_check = 0.0
    except (ImportError, AttributeError):
        pass
    try:
        import omega.entity.engine as ee
        ee._cache_ts = 0.0
    except (ImportError, AttributeError):
        pass
    try:
        from omega import advisor
        advisor._session_dedup.clear()
        advisor._file_cooldowns.clear()
    except (ImportError, AttributeError):
        pass
    yield
    try:
        from omega.server import hook_server
        hook_server._debounce_state.reset()
        hook_server._last_reminder_check = 0.0
    except (ImportError, AttributeError):
        pass
    try:
        import omega.entity.engine as ee
        ee._cache_ts = 0.0
    except (ImportError, AttributeError):
        pass
    try:
        from omega import advisor
        advisor._session_dedup.clear()
        advisor._file_cooldowns.clear()
    except (ImportError, AttributeError):
        pass


@pytest.fixture
def _reset_bridge(tmp_omega_dir):
    """Reset the bridge singleton so each test gets a fresh store.

    Centralized definition: test modules can use this via
    @pytest.mark.usefixtures("_reset_bridge") or request it directly.
    Modules that need autouse behavior can define a local autouse fixture
    that depends on this one.
    """
    from omega.bridge import reset_memory

    reset_memory()
    yield
    reset_memory()


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
    mgr = CoordinationManager(db_path=db_path, cloud_sync=False)
    yield mgr
    mgr.close()
