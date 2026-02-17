"""Tests for OMEGA HTTP server (Streamable HTTP transport)."""

import stat
import pytest
from unittest.mock import MagicMock, patch

# starlette is a transitive dep of mcp[server], not installed in base CI
pytest.importorskip("starlette", reason="starlette not installed (requires 'server' extra)")

from omega.server.http_server import create_http_app, get_or_create_api_key


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_server():
    """Create a mock MCP Server object for testing."""
    server = MagicMock()
    server.name = "omega-memory"
    return server


@pytest.fixture
def app(mock_server):
    """Create an HTTP app with auth disabled."""
    return create_http_app(mock_server, api_key=None)


@pytest.fixture
def app_with_auth(mock_server):
    """Create an HTTP app with auth enabled."""
    return create_http_app(mock_server, api_key="test-secret-key")


# ============================================================================
# App creation tests
# ============================================================================

def test_create_http_app(mock_server):
    """create_http_app returns a Starlette app with the expected routes."""
    app = create_http_app(mock_server)
    route_paths = {r.path for r in app.routes}
    assert "/mcp" in route_paths
    assert "/health" in route_paths
    assert "/.well-known/mcp.json" in route_paths


def test_create_http_app_with_auth(mock_server):
    """create_http_app accepts an api_key parameter."""
    app = create_http_app(mock_server, api_key="my-key")
    assert app is not None


# ============================================================================
# Health endpoint tests
# ============================================================================

def test_health_endpoint(app):
    """GET /health returns 200 with status ok."""
    from starlette.testclient import TestClient

    with TestClient(app) as client:
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["server"] == "omega-memory"


# ============================================================================
# Server card endpoint tests
# ============================================================================

def test_server_card_endpoint(app):
    """GET /.well-known/mcp.json returns valid server card."""
    from starlette.testclient import TestClient

    with TestClient(app) as client:
        resp = client.get("/.well-known/mcp.json")
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "omega-memory"
        assert "version" in data
        assert "transports" in data
        assert isinstance(data["tools_count"], int)
        assert data["tools_count"] > 0
        transport_types = [t["type"] for t in data["transports"]]
        assert "streamable-http" in transport_types
        assert "stdio" in transport_types


# ============================================================================
# Auth tests
# ============================================================================

def test_mcp_endpoint_requires_auth(app_with_auth):
    """POST /mcp without API key returns 401 when auth is enabled."""
    from starlette.testclient import TestClient

    with TestClient(app_with_auth) as client:
        resp = client.post("/mcp/", json={"jsonrpc": "2.0", "method": "tools/list", "id": 1})
        assert resp.status_code == 401
        assert resp.json()["error"] == "Unauthorized"


def test_mcp_endpoint_wrong_key(app_with_auth):
    """POST /mcp with wrong API key returns 401."""
    from starlette.testclient import TestClient

    with TestClient(app_with_auth) as client:
        resp = client.post(
            "/mcp/",
            json={"jsonrpc": "2.0", "method": "tools/list", "id": 1},
            headers={"X-API-Key": "wrong-key"},
        )
        assert resp.status_code == 401


def test_mcp_endpoint_auth_via_query_param(app_with_auth):
    """POST /mcp with api_key query param passes auth (not 401)."""
    from starlette.testclient import TestClient

    with TestClient(app_with_auth, raise_server_exceptions=False) as client:
        resp = client.post(
            "/mcp/?api_key=test-secret-key",
            json={"jsonrpc": "2.0", "method": "tools/list", "id": 1},
        )
        # Auth passed — should NOT be 401. The mock server can't handle MCP
        # protocol so we may get a 500, but the auth layer let it through.
        assert resp.status_code != 401


def test_mcp_endpoint_auth_via_header(app_with_auth):
    """POST /mcp with X-API-Key header passes auth (not 401)."""
    from starlette.testclient import TestClient

    with TestClient(app_with_auth, raise_server_exceptions=False) as client:
        resp = client.post(
            "/mcp/",
            json={"jsonrpc": "2.0", "method": "tools/list", "id": 1},
            headers={"X-API-Key": "test-secret-key"},
        )
        assert resp.status_code != 401


def test_no_auth_mode(app):
    """When api_key=None, requests pass through without auth."""
    from starlette.testclient import TestClient

    with TestClient(app, raise_server_exceptions=False) as client:
        resp = client.post(
            "/mcp/",
            json={"jsonrpc": "2.0", "method": "tools/list", "id": 1},
        )
        # No auth configured — should NOT be 401
        assert resp.status_code != 401


# ============================================================================
# API key management tests
# ============================================================================

def test_api_key_generation(tmp_path):
    """get_or_create_api_key creates a file with correct permissions."""
    key_path = tmp_path / "api_key"
    with patch("omega.server.http_server.API_KEY_PATH", key_path):
        key = get_or_create_api_key()
        assert len(key) > 20  # URL-safe token is ~43 chars for 32 bytes
        assert key_path.exists()
        mode = key_path.stat().st_mode
        assert mode & stat.S_IRWXG == 0  # no group access
        assert mode & stat.S_IRWXO == 0  # no other access


def test_api_key_persistence(tmp_path):
    """Second call returns the same key."""
    key_path = tmp_path / "api_key"
    with patch("omega.server.http_server.API_KEY_PATH", key_path):
        key1 = get_or_create_api_key()
        key2 = get_or_create_api_key()
        assert key1 == key2


def test_api_key_reads_existing(tmp_path):
    """get_or_create_api_key reads an existing key file."""
    key_path = tmp_path / "api_key"
    key_path.write_text("my-custom-key\n")
    with patch("omega.server.http_server.API_KEY_PATH", key_path):
        key = get_or_create_api_key()
        assert key == "my-custom-key"
