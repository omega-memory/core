"""OMEGA MCP Server -- Standalone stdio-based MCP server for Claude Code."""

import atexit
import asyncio
import collections
import logging
import os
import sys
import time

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from omega.server.tool_schemas import TOOL_SCHEMAS as _CORE_SCHEMAS
from omega.server.handlers import HANDLERS as _CORE_HANDLERS

# Start with core memory tools
TOOL_SCHEMAS = list(_CORE_SCHEMAS)
HANDLERS = dict(_CORE_HANDLERS)

# Built-in optional modules (coordination, router, profile, knowledge, entity).
# Each is tried in turn; missing modules are silently skipped.
_BUILTIN_MODULES = [
    ("omega.server.coord_schemas", "COORD_TOOL_SCHEMAS", "omega.server.coord_handlers", "COORD_HANDLERS"),
    ("omega.router.tool_schemas", "ROUTER_TOOL_SCHEMAS", "omega.router.handlers", "ROUTER_HANDLERS"),
    ("omega.profile.tool_schemas", "PROFILE_TOOL_SCHEMAS", "omega.profile.handlers", "PROFILE_HANDLERS"),
    ("omega.knowledge.tool_schemas", "KNOWLEDGE_TOOL_SCHEMAS", "omega.knowledge.handlers", "KNOWLEDGE_HANDLERS"),
    ("omega.entity.tool_schemas", "ENTITY_TOOL_SCHEMAS", "omega.entity.handlers", "ENTITY_HANDLERS"),
]

import importlib

for _schema_mod, _schema_attr, _handler_mod, _handler_attr in _BUILTIN_MODULES:
    try:
        _sm = importlib.import_module(_schema_mod)
        _hm = importlib.import_module(_handler_mod)
        TOOL_SCHEMAS = TOOL_SCHEMAS + getattr(_sm, _schema_attr)
        HANDLERS = {**HANDLERS, **getattr(_hm, _handler_attr)}
    except ImportError:
        pass

# Discover external plugins (e.g. omega-pro)
from omega.plugins import discover_plugins

_discovered_plugins = discover_plugins()
for _plugin in _discovered_plugins:
    if _plugin.TOOL_SCHEMAS:
        TOOL_SCHEMAS = TOOL_SCHEMAS + _plugin.TOOL_SCHEMAS
    if _plugin.HANDLERS:
        HANDLERS = {**HANDLERS, **_plugin.HANDLERS}

# Wire plugin retrieval profiles and score modifiers to SQLiteStore (lazy)
def _wire_plugin_retrieval():
    """Register plugin retrieval profiles and score modifiers on the store."""
    try:
        from omega.bridge import _get_store
        store = _get_store()
        for plugin in _discovered_plugins:
            if getattr(plugin, "RETRIEVAL_PROFILES", None):
                store.register_plugin_profiles(plugin.RETRIEVAL_PROFILES)
            for modifier in getattr(plugin, "SCORE_MODIFIERS", []):
                store.register_score_modifier(modifier)
    except Exception:
        pass  # Store not ready yet; profiles will use built-in defaults

# Idle watchdog: exit after this many seconds without a tool call.
# Override with OMEGA_IDLE_TIMEOUT env var. 0 = disabled.
_IDLE_TIMEOUT = int(os.environ.get("OMEGA_IDLE_TIMEOUT", "3600"))
_last_activity: float = time.monotonic()


def _close_on_exit():
    """Close SQLite store when the MCP server process exits."""
    try:
        from omega.bridge import _close_store

        _close_store()
    except Exception:
        pass


atexit.register(_close_on_exit)

logger = logging.getLogger("omega.server")

server = Server("omega-memory")

# ---------------------------------------------------------------------------
# Rate limiting — sliding-window counters (no new deps)
# ---------------------------------------------------------------------------
_GLOBAL_RATE_LIMIT = int(os.environ.get("OMEGA_RATE_LIMIT_GLOBAL", "300"))  # per minute
_WRITE_RATE_LIMIT = int(os.environ.get("OMEGA_RATE_LIMIT_WRITE", "60"))  # per minute
_RATE_WINDOW_S = 60.0

_global_timestamps: collections.deque = collections.deque()
_write_timestamps: collections.deque = collections.deque()

_WRITE_TOOLS = frozenset({
    "omega_store", "omega_checkpoint", "omega_remind",
    "omega_profile_set", "omega_entity_create", "omega_entity_update",
    "omega_ingest_document", "omega_task_create",
    "omega_file_claim", "omega_branch_claim",
    "omega_send_message", "omega_intent_announce",
})


def _check_rate_limit(tool_name: str) -> str | None:
    """Return an error message if rate limit exceeded, else None."""
    now = time.monotonic()
    cutoff = now - _RATE_WINDOW_S

    # Prune expired entries
    while _global_timestamps and _global_timestamps[0] < cutoff:
        _global_timestamps.popleft()

    if len(_global_timestamps) >= _GLOBAL_RATE_LIMIT:
        return f"Rate limit exceeded: {_GLOBAL_RATE_LIMIT} calls/min globally. Try again shortly."

    _global_timestamps.append(now)

    # Write-tool tier
    if tool_name in _WRITE_TOOLS:
        while _write_timestamps and _write_timestamps[0] < cutoff:
            _write_timestamps.popleft()
        if len(_write_timestamps) >= _WRITE_RATE_LIMIT:
            return f"Rate limit exceeded: {_WRITE_RATE_LIMIT} write calls/min. Try again shortly."
        _write_timestamps.append(now)

    return None


@server.list_tools()
async def list_tools() -> list[Tool]:
    """Return all OMEGA tools."""
    return [
        Tool(
            name=schema["name"],
            description=schema["description"],
            inputSchema=schema["inputSchema"],
        )
        for schema in TOOL_SCHEMAS
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Dispatch tool call to the appropriate handler."""
    global _last_activity
    _last_activity = time.monotonic()

    # Rate limiting
    rate_err = _check_rate_limit(name)
    if rate_err:
        return [TextContent(type="text", text=rate_err)]

    handler = HANDLERS.get(name)
    if not handler:
        return [TextContent(type="text", text=f"Unknown tool: {name}")]

    try:
        result = await handler(arguments)
        # Extract text from MCP response format
        content_list = result.get("content", [{}])
        text = content_list[0].get("text", str(result)) if content_list else str(result)
        return [TextContent(type="text", text=text)]
    except Exception as e:
        logger.error("Tool %s failed: %s", name, e)
        return [TextContent(type="text", text=f"Error in {name}: {e}")]


async def _idle_watchdog():
    """Exit the process if no tool call has been received within the timeout."""
    while True:
        await asyncio.sleep(30)
        idle = time.monotonic() - _last_activity
        if idle >= _IDLE_TIMEOUT:
            logger.warning("Idle for %.0fs (limit %ds), shutting down.", idle, _IDLE_TIMEOUT)
            _close_on_exit()
            os._exit(0)


async def _socket_watchdog():
    """Re-create the hook socket if it gets deleted by another process."""
    from omega.server.hook_server import SOCK_PATH, start_hook_server

    while True:
        await asyncio.sleep(15)
        if not SOCK_PATH.exists():
            logger.warning("Hook socket deleted, re-creating...")
            await start_hook_server()


_coord_tick_count = 0


def _run_coordination_tick():
    """Sync helper for periodic coordination maintenance."""
    global _coord_tick_count
    _coord_tick_count += 1
    try:
        from omega.coordination import get_manager
        from omega.server.hook_server import (
            _last_deadlock_push,
            DEADLOCK_PUSH_DEBOUNCE_S,
            _agent_nickname,
        )

        mgr = get_manager()

        # Stale cleanup — internally debounced to 5 min
        try:
            mgr._maybe_clean_stale()
        except Exception:
            pass

        # Every 5th tick (~5 min): deadlock detection + push
        if _coord_tick_count % 5 == 0:
            try:
                cycles = mgr.detect_deadlocks()
                if cycles:
                    now_dl = time.monotonic()
                    for cycle in cycles[:2]:
                        cycle_key = str(hash(tuple(sorted(cycle[:-1]))))
                        if cycle_key not in _last_deadlock_push or now_dl - _last_deadlock_push[cycle_key] >= DEADLOCK_PUSH_DEBOUNCE_S:
                            _last_deadlock_push[cycle_key] = now_dl
                            cycle_str = " -> ".join(_agent_nickname(s) for s in cycle)
                            for peer in set(cycle[:-1]):
                                try:
                                    mgr.send_message(
                                        from_session=peer,
                                        subject=f"[DEADLOCK] Circular wait: {cycle_str}",
                                        to_session=peer,
                                        msg_type="inform",
                                        ttl_minutes=30,
                                    )
                                except Exception:
                                    pass
            except Exception:
                pass
    except Exception:
        pass  # All fail-open


async def _coordination_loop():
    """Periodic coordination maintenance — runs even during idle."""
    loop = asyncio.get_running_loop()
    while True:
        await asyncio.sleep(60)
        try:
            await loop.run_in_executor(None, _run_coordination_tick)
        except Exception:
            pass  # All fail-open


async def main():
    """Entry point for the OMEGA MCP server."""
    logging.basicConfig(level=logging.WARNING, stream=sys.stderr)
    logger.info("Starting OMEGA MCP server...")

    # Start UDS hook server for fast hook dispatch
    from omega.server.hook_server import start_hook_server, stop_hook_server

    hook_srv = await start_hook_server()

    # Prewarm embedding model in background — hides 200-500ms ONNX load
    # behind session startup rather than blocking the first user query.
    async def _prewarm():
        try:
            from omega.graphs import preload_embedding_model_async
            await preload_embedding_model_async()
        except Exception:
            pass  # Non-fatal — lazy-load on first query as fallback

    _prewarm_task = asyncio.create_task(_prewarm())

    # Wire plugin retrieval profiles/modifiers to SQLiteStore
    _wire_plugin_retrieval()

    # Start idle watchdog (unless disabled).
    # IMPORTANT: Save reference — unref'd tasks get silently GC'd by asyncio.
    if _IDLE_TIMEOUT > 0:
        _watchdog_task = asyncio.create_task(_idle_watchdog())

    # Socket watchdog — re-creates hook.sock if deleted by another session's stop
    _sock_watchdog_task = asyncio.create_task(_socket_watchdog())

    # Background coordination loop — stale cleanup + deadlock detection even during idle
    _coord_loop_task = asyncio.create_task(_coordination_loop())

    try:
        async with stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                server.create_initialization_options(),
            )
    finally:
        await stop_hook_server(hook_srv)


if __name__ == "__main__":
    asyncio.run(main())
