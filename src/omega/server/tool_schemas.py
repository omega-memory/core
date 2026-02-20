"""OMEGA MCP Tool Schemas -- 12 tools for memory management.

Consolidated into 12 action-discriminated composites.
All original capabilities preserved; low-frequency operations grouped by intent.
"""

TOOL_SCHEMAS = [
    {
        "name": "omega_store",
        "description": "Store a memory with optional type and metadata. Use when the user says 'remember this' or for programmatic capture (decisions, lessons, errors). Defaults to type 'memory' if event_type is omitted.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "Memory content (also accepts 'text' as alias)"},
                "text": {"type": "string", "description": "Alias for content"},
                "event_type": {
                    "type": "string",
                    "description": "Type: memory (default), session_summary, task_completion, error_pattern, lesson_learned, decision, user_preference",
                },
                "metadata": {"type": "object", "description": "Additional metadata"},
                "session_id": {"type": "string"},
                "project": {"type": "string"},
                "priority": {
                    "type": "integer",
                    "description": "Memory priority 1-5 (5=highest). Auto-set from event type if omitted.",
                    "minimum": 1,
                    "maximum": 5,
                },
                "entity_id": {
                    "type": "string",
                    "description": "Scope this memory to an entity (e.g., 'acme'). Omit for unscoped.",
                },
                "agent_type": {
                    "type": "string",
                    "description": "Agent type for sub-agent memory scoping (e.g., 'code-reviewer', 'test-runner').",
                },
            },
            "required": ["content"],
        },
    },
    {
        "name": "omega_query",
        "description": "Search memories. Modes: 'semantic' (default) for meaning-based search, 'phrase' for exact substring match, 'timeline' for recent memories grouped by day.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query (or exact phrase when mode='phrase'). Not required for mode='timeline'."},
                "mode": {
                    "type": "string",
                    "enum": ["semantic", "phrase", "timeline"],
                    "description": "Search mode: 'semantic' (default), 'phrase' for exact match, 'timeline' for recent memories by day",
                },
                "limit": {"type": "integer", "default": 10},
                "event_type": {"type": "string", "description": "Filter by event type"},
                "project": {"type": "string"},
                "session_id": {"type": "string"},
                "context_file": {"type": "string", "description": "Current file being edited (boosts results)"},
                "context_tags": {"type": "array", "items": {"type": "string"}, "description": "Context tags for boosting"},
                "filter_tags": {"type": "array", "items": {"type": "string"}, "description": "Hard filter: ALL tags must match (AND logic)"},
                "temporal_range": {"type": "array", "items": {"type": "string"}, "minItems": 2, "maxItems": 2, "description": "[start_iso, end_iso] date range filter"},
                "entity_id": {"type": "string", "description": "Filter to entity. Omit for all."},
                "agent_type": {"type": "string", "description": "Filter to agent type. Omit for all."},
                "case_sensitive": {"type": "boolean", "description": "Case-sensitive (only for mode='phrase', default false)", "default": False},
                "days": {"type": "integer", "description": "Days to look back (only for mode='timeline', default 7)", "default": 7},
                "limit_per_day": {"type": "integer", "description": "Max per day (only for mode='timeline', default 10)", "default": 10},
            },
        },
    },
    {
        "name": "omega_welcome",
        "description": "Session startup briefing. Call at the beginning of every session to load recent context, active reminders, and user profile. Returns what the agent needs to continue where the last session left off.",
        "inputSchema": {"type": "object", "properties": {"session_id": {"type": "string"}, "project": {"type": "string"}}},
    },
    {
        "name": "omega_protocol",
        "description": "Retrieve your operating rules and behavioral guidelines for this session. Returns context-sensitive instructions covering memory usage, coordination, reminders, and workflow. Call after omega_welcome at session start, or on-demand for a specific section.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "section": {"type": "string", "description": "Section: 'memory', 'coordination', 'coordination_gate', 'teamwork', 'context', 'reminders', 'diagnostics', 'entity', 'heuristics', 'git', 'what_next'. Groups: 'solo', 'multi_agent', 'full', 'minimal'."},
                "project": {"type": "string", "description": "Project path for context-sensitive rules."},
            },
        },
    },
    {
        "name": "omega_lessons",
        "description": "Retrieve lessons learned from past sessions to avoid repeating mistakes. Use before starting a task to check for known pitfalls. Results ranked by verification count and access frequency. Supports cross-project search.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "task": {"type": "string", "description": "Task description for relevance filtering"},
                "project_path": {"type": "string", "description": "Project scope"},
                "limit": {"type": "integer", "description": "Max lessons (default 5)", "default": 5},
                "cross_project": {"type": "boolean", "description": "Search across all projects (default false)", "default": False},
                "exclude_project": {"type": "string", "description": "Project to exclude (with cross_project=true)"},
                "exclude_session": {"type": "string", "description": "Session ID to exclude"},
                "agent_type": {"type": "string", "description": "Filter to agent type. Omit for all."},
            },
        },
    },
    {
        "name": "omega_checkpoint",
        "description": "Save a task checkpoint: captures current plan, progress, files touched, decisions, and key context. Enables seamless session continuity.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "task_title": {"type": "string", "description": "Brief title of the current task"},
                "plan": {"type": "string", "description": "Current plan or goals"},
                "progress": {"type": "string", "description": "What's been completed, in progress, remaining"},
                "files_touched": {"type": "object", "description": "Map of file paths to change summaries", "additionalProperties": {"type": "string"}},
                "decisions": {"type": "array", "items": {"type": "string"}, "description": "Key technical decisions"},
                "key_context": {"type": "string", "description": "Critical context for continuation"},
                "next_steps": {"type": "string", "description": "What to do next"},
                "session_id": {"type": "string"},
                "project": {"type": "string"},
            },
            "required": ["task_title", "progress"],
        },
    },
    {
        "name": "omega_resume_task",
        "description": "Resume a previously checkpointed task. Retrieves the latest checkpoint with full plan, progress, files, decisions, and next steps.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "task_title": {"type": "string", "description": "Title of the task to resume (semantic search)"},
                "project": {"type": "string", "description": "Project path to filter checkpoints"},
                "verbosity": {"type": "string", "enum": ["full", "summary", "minimal"], "description": "full=everything, summary=plan+progress+next, minimal=next steps only"},
                "limit": {"type": "integer", "description": "Number of checkpoints to retrieve (default 1)"},
            },
        },
    },
    {
        "name": "omega_memory",
        "description": "Manage a specific memory by ID: edit its content, delete it, mark it as helpful/unhelpful/outdated, find similar memories, or traverse relationship edges. Use when acting on an individual memory rather than searching broadly.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["edit", "delete", "feedback", "similar", "traverse"], "description": "Operation to perform"},
                "memory_id": {"type": "string", "description": "Memory node ID (required for all actions)"},
                "new_content": {"type": "string", "description": "New content (only for action='edit')"},
                "rating": {"type": "string", "description": "helpful, unhelpful, or outdated (only for action='feedback')"},
                "reason": {"type": "string", "description": "Optional explanation for feedback"},
                "limit": {"type": "integer", "description": "Max results for similar (default 5)", "default": 5},
                "max_hops": {"type": "integer", "description": "Traversal depth 1-5 (default 2, only for action='traverse')", "default": 2},
                "min_weight": {"type": "number", "description": "Min edge weight 0.0-1.0 (default 0.0, only for action='traverse')", "default": 0.0},
            },
            "required": ["action", "memory_id"],
        },
    },
    {
        "name": "omega_profile",
        "description": "Read or update the user's persistent profile (name, preferences, working style) or list all stored preferences. The profile persists across sessions and informs agent behavior. Default action is 'read'.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["read", "update", "list_preferences"], "description": "read (default), update, or list_preferences", "default": "read"},
                "update": {"type": "object", "description": "Profile fields to merge (only for action='update')"},
            },
        },
    },
    {
        "name": "omega_remind",
        "description": "Manage time-based reminders: set new reminders, list active ones, or dismiss by ID.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["set", "list", "dismiss"], "description": "set (default), list, or dismiss", "default": "set"},
                "text": {"type": "string", "description": "What to be reminded about (for action='set')"},
                "duration": {"type": "string", "description": "When to remind, e.g. '1h', '30m', '2d' (for action='set')"},
                "context": {"type": "string", "description": "Optional context (for action='set')"},
                "session_id": {"type": "string"},
                "project": {"type": "string"},
                "status": {"type": "string", "enum": ["pending", "fired", "dismissed", "all"], "description": "Filter (for action='list')"},
                "reminder_id": {"type": "string", "description": "Reminder ID (for action='dismiss')"},
            },
        },
    },
    {
        "name": "omega_maintain",
        "description": "System housekeeping for the memory store. Use 'health' to check database size and integrity, 'consolidate' to prune stale memories, 'compact' to merge near-duplicates, 'backup'/'restore' for data safety, 'clear_session' to purge a session's data. Call periodically or when memory grows large.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["health", "consolidate", "compact", "backup", "restore", "clear_session"], "description": "Maintenance operation"},
                "warn_mb": {"type": "number", "description": "Warning threshold MB (health, default 350)", "default": 350},
                "critical_mb": {"type": "number", "description": "Critical threshold MB (health, default 800)", "default": 800},
                "max_nodes": {"type": "integer", "description": "Max expected nodes (health, default 10000)", "default": 10000},
                "prune_days": {"type": "integer", "description": "Prune zero-access older than N days (consolidate, default 14)", "default": 14},
                "max_summaries": {"type": "integer", "description": "Max session summaries (consolidate, default 50)", "default": 50},
                "event_type": {"type": "string", "description": "Type to compact (compact, default lesson_learned)", "default": "lesson_learned"},
                "similarity_threshold": {"type": "number", "description": "Jaccard similarity 0.0-1.0 (compact, default 0.6)", "default": 0.6},
                "min_cluster_size": {"type": "integer", "description": "Min cluster size (compact, default 3)", "default": 3},
                "dry_run": {"type": "boolean", "description": "Preview only (compact, default false)", "default": False},
                "filepath": {"type": "string", "description": "File path (backup/restore)"},
                "clear_existing": {"type": "boolean", "description": "Clear before restore (default true)", "default": True},
                "session_id": {"type": "string", "description": "Session to purge (clear_session)"},
            },
            "required": ["action"],
        },
    },
    {
        "name": "omega_stats",
        "description": "View analytics about stored memories: breakdown by type, per-session statistics, weekly activity digest, or access rate trends. Use to understand memory growth, usage patterns, and health over time.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["types", "sessions", "digest", "access_rate"], "description": "Which stats to retrieve"},
                "days": {"type": "integer", "description": "Days for digest (default 7)", "default": 7},
            },
            "required": ["action"],
        },
    },
]
