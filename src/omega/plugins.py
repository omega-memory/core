"""OMEGA Plugin Interface — extensible discovery for commercial modules.

Core provides entry-point-based plugin discovery. Commercial packages
register via ``[project.entry-points."omega.plugins"]`` in their pyproject.toml.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

logger = logging.getLogger("omega.plugins")


class OmegaPlugin:
    """Base class for OMEGA plugins.

    Subclasses should populate these class-level attributes:

    - ``TOOL_SCHEMAS``: list of MCP tool schema dicts
    - ``HANDLERS``: dict mapping tool name → async handler function
    - ``HOOK_HANDLERS``: dict mapping hook name → sync handler function
    - ``CLI_COMMANDS``: list of (name, setup_func) tuples where setup_func(subparsers)
      registers an argparse subparser; the plugin class should also provide a
      ``cmd_{name}(args)`` method as the command handler
    - ``HOOKS_JSON``: dict matching the hooks.json manifest format (optional)
    - ``RETRIEVAL_PROFILES``: dict mapping event_type → (vec, text, word, ctx, graph) phase weights
    - ``SCORE_MODIFIERS``: list of fn(node_id, score, metadata) → score callables
    """

    TOOL_SCHEMAS: list[dict[str, Any]] = []
    HANDLERS: dict[str, Callable] = {}
    HOOK_HANDLERS: dict[str, Callable] = {}
    CLI_COMMANDS: list[tuple[str, Callable]] = []
    HOOKS_JSON: dict[str, Any] = {}
    RETRIEVAL_PROFILES: dict[str, tuple] = {}
    SCORE_MODIFIERS: list[Callable] = []


def discover_plugins() -> list[OmegaPlugin]:
    """Discover and instantiate all registered OMEGA plugins.

    Looks up ``omega.plugins`` entry-point group via importlib.metadata.
    Each entry point should reference a class that inherits from OmegaPlugin.
    Returns an empty list if no plugins are installed.
    """
    plugins: list[OmegaPlugin] = []
    try:
        from importlib.metadata import entry_points

        eps = entry_points(group="omega.plugins")
        for ep in eps:
            try:
                plugin_cls = ep.load()
                if isinstance(plugin_cls, type) and issubclass(plugin_cls, OmegaPlugin):
                    plugins.append(plugin_cls())
                elif isinstance(plugin_cls, OmegaPlugin):
                    plugins.append(plugin_cls)
                else:
                    logger.warning("Plugin %s is not an OmegaPlugin subclass, skipping", ep.name)
            except Exception as e:
                logger.warning("Failed to load plugin %s: %s", ep.name, e)
    except Exception as e:
        logger.debug("Plugin discovery unavailable: %s", e)
    return plugins
