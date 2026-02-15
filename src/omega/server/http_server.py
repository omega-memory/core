"""OMEGA HTTP Server — Streamable HTTP transport for the MCP server.

Wraps the existing stdio-based MCP server in a Starlette ASGI app using
the MCP SDK's StreamableHTTPSessionManager. This enables:
- Remote access (Docker, multi-IDE, mobile via Tailscale)
- Smithery.ai marketplace listing
- Native HTTP without external mcp-proxy dependency

Dependencies (starlette, uvicorn) are already transitive deps of mcp>=1.0.0.
"""

import contextlib
import secrets
from collections.abc import AsyncIterator
from pathlib import Path

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Mount, Route
from starlette.types import Receive, Scope, Send

from mcp.server.streamable_http_manager import StreamableHTTPSessionManager

API_KEY_PATH = Path.home() / ".omega" / "api_key"


def get_or_create_api_key() -> str:
    """Load API key from ~/.omega/api_key, or generate one."""
    if API_KEY_PATH.exists():
        return API_KEY_PATH.read_text().strip()
    key = secrets.token_urlsafe(32)
    API_KEY_PATH.parent.mkdir(parents=True, exist_ok=True)
    API_KEY_PATH.write_text(key + "\n")
    API_KEY_PATH.chmod(0o600)
    return key


def create_http_app(server, api_key: str | None = None) -> Starlette:
    """Create a Starlette ASGI app wrapping the MCP server.

    Args:
        server: The MCP Server instance from mcp_server.py.
        api_key: Optional API key for authentication. None disables auth.
    """
    session_manager = StreamableHTTPSessionManager(
        app=server,
        json_response=True,
        stateless=True,
    )

    async def mcp_asgi_app(scope: Scope, receive: Receive, send: Send) -> None:
        """ASGI app for the /mcp endpoint — delegates to StreamableHTTPSessionManager."""
        if api_key:
            request = Request(scope, receive)
            provided = request.headers.get("x-api-key") or request.query_params.get("api_key")
            if provided != api_key:
                response = JSONResponse({"error": "Unauthorized"}, status_code=401)
                await response(scope, receive, send)
                return
        await session_manager.handle_request(scope, receive, send)

    async def health(request: Request):
        return JSONResponse({"status": "ok", "server": "omega-memory"})

    async def server_card(request: Request):
        from omega import __version__
        from omega.server.mcp_server import TOOL_SCHEMAS

        return JSONResponse({
            "name": "omega-memory",
            "version": __version__,
            "description": "Persistent memory for AI coding agents",
            "homepage": "https://omegamax.co",
            "repository": "https://github.com/omega-memory/omega-memory",
            "transports": [
                {"type": "streamable-http", "url": "/mcp"},
                {"type": "stdio", "command": "python3 -m omega.server.mcp_server"},
            ],
            "tools_count": len(TOOL_SCHEMAS),
        })

    @contextlib.asynccontextmanager
    async def lifespan(app: Starlette) -> AsyncIterator[None]:
        async with session_manager.run():
            yield

    app = Starlette(
        routes=[
            Mount("/mcp", app=mcp_asgi_app),
            Route("/health", endpoint=health),
            Route("/.well-known/mcp.json", endpoint=server_card),
            Route("/.well-known/mcp/server-card.json", endpoint=server_card),
        ],
        lifespan=lifespan,
    )
    return app


async def run_http(host: str, port: int, api_key: str | None) -> None:
    """Import existing server, create HTTP app, run uvicorn."""
    import uvicorn

    from omega.server.mcp_server import server, _wire_plugin_retrieval
    from omega.server.hook_server import start_hook_server

    await start_hook_server()
    _wire_plugin_retrieval()

    app = create_http_app(server, api_key=api_key)
    config = uvicorn.Config(app, host=host, port=port, log_level="info")
    srv = uvicorn.Server(config)
    await srv.serve()
