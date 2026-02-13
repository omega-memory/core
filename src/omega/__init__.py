"""OMEGA — Persistent memory for AI coding agents."""

__version__ = "0.8.0"

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
)

__all__ = [
    "SQLiteStore",
    "remember",
    "store",
    "query",
    "query_structured",
    "phrase_search",
    "welcome",
    "status",
    "auto_capture",
    "delete_memory",
    "edit_memory",
    "find_similar_memories",
    "timeline",
    "consolidate",
    "compact",
    "traverse",
    "check_health",
    "type_stats",
    "session_stats",
    "export_memories",
    "import_memories",
    "__version__",
]
