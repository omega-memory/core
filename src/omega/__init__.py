"""OMEGA — Persistent memory for AI coding agents.

Direct Python API -- no MCP server required::

    from omega import store, query, remember
    store("Always use TypeScript strict mode", "user_preference")
    results = query("TypeScript preferences")

For full Claude Code integration (MCP tools, auto-capture, coordination),
install with: ``pip install omega-memory[server]``
"""

__version__ = "0.9.3"

from omega.sqlite_store import SQLiteStore
from omega.bridge import (
    remember,
    store,
    query,
    query_structured,
    phrase_search,
    welcome,
    status,
    auto_capture,
    delete_memory,
    edit_memory,
    find_similar_memories,
    timeline,
    consolidate,
    compact,
    traverse,
    check_health,
    type_stats,
    session_stats,
    export_memories,
    import_memories,
    batch_store,
    record_feedback,
    deduplicate,
    get_session_context,
    get_activity_summary,
    create_reminder,
    list_reminders,
    dismiss_reminder,
    get_due_reminders,
)

__all__ = [
    "SQLiteStore",
    # Core CRUD
    "store",
    "remember",
    "query",
    "query_structured",
    "phrase_search",
    "batch_store",
    "delete_memory",
    "edit_memory",
    # Session & lifecycle
    "welcome",
    "status",
    "auto_capture",
    "get_session_context",
    "get_activity_summary",
    # Maintenance
    "consolidate",
    "compact",
    "deduplicate",
    "check_health",
    "record_feedback",
    # Navigation
    "find_similar_memories",
    "timeline",
    "traverse",
    "type_stats",
    "session_stats",
    # Import/export
    "export_memories",
    "import_memories",
    # Reminders
    "create_reminder",
    "list_reminders",
    "dismiss_reminder",
    "get_due_reminders",
    # Meta
    "__version__",
]
