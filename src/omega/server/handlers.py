"""
OMEGA MCP Handlers -- Maps tool names to async handler functions.

Each handler delegates to omega.bridge for actual operations and returns
MCP-compatible response dicts.
"""

import logging
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger("omega.server.handlers")


def _clamp_int(value, default: int, min_val: int = 1, max_val: int = 10000) -> int:
    """Clamp a numeric argument to safe bounds."""
    try:
        v = int(value)
        return max(min_val, min(v, max_val))
    except (TypeError, ValueError):
        return default


# Safe directory for export/import operations
_SAFE_EXPORT_DIR = Path.home() / ".omega"


# ============================================================================
# Response Helpers
# ============================================================================


def mcp_response(text: str) -> dict:
    """Build a successful MCP response."""
    return {"content": [{"type": "text", "text": str(text)}]}


def mcp_error(text: str) -> dict:
    """Build an error MCP response."""
    return {"content": [{"type": "text", "text": f"Error: {text}"}], "isError": True}


# ============================================================================
# Handler: omega_store (also handles omega_remember as alias)
# ============================================================================


async def handle_omega_store(arguments: dict) -> dict:
    """Store a memory with optional type and metadata.

    Accepts 'text' as alias for 'content' for backward compat with omega_remember.
    Defaults event_type to 'memory' when not provided.
    """
    content = arguments.get("content", "").strip()
    # Support 'text' as alias for 'content' (backward compat with omega_remember)
    if not content:
        content = arguments.get("text", "").strip()
    if not content:
        return mcp_error("content (or text) is required")

    event_type = arguments.get("event_type", "memory")
    metadata = arguments.get("metadata", {})
    session_id = arguments.get("session_id")
    project = arguments.get("project") or (metadata or {}).get("project")
    entity_id = arguments.get("entity_id")
    agent_type = arguments.get("agent_type")

    # Wire through priority if provided
    priority = arguments.get("priority")
    if priority is not None:
        try:
            priority = max(1, min(5, int(priority)))
            metadata = dict(metadata or {})
            metadata["priority"] = priority
        except (TypeError, ValueError):
            pass

    try:
        from omega.bridge import store

        result = store(
            content=content,
            event_type=event_type,
            metadata=metadata,
            session_id=session_id,
            project=project,
            entity_id=entity_id,
            agent_type=agent_type,
        )
        return mcp_response(result)
    except Exception as e:
        logger.error("omega_store failed: %s", e)
        import traceback
        tb = traceback.format_exc()
        logger.error("omega_store traceback: %s", tb)
        return mcp_error(f"Failed to store memory: {e}")


# ============================================================================
# Handler: omega_query
# ============================================================================


async def handle_omega_query(arguments: dict) -> dict:
    """Search memories — semantic (default) or exact phrase match."""
    query_text = arguments.get("query", "").strip()
    if not query_text:
        return mcp_error("query is required")

    mode = arguments.get("mode", "semantic")

    # Phrase mode — delegate to bridge.phrase_search
    if mode == "phrase":
        limit = _clamp_int(arguments.get("limit", 10), default=10, max_val=1000)
        event_type = arguments.get("event_type")
        project = arguments.get("project")
        case_sensitive = arguments.get("case_sensitive", False)
        try:
            from omega.bridge import phrase_search

            result = phrase_search(
                phrase=query_text,
                limit=limit,
                event_type=event_type,
                project=project,
                case_sensitive=case_sensitive,
            )
            return mcp_response(result)
        except Exception as e:
            logger.error("omega_query (phrase) failed: %s", e)
            return mcp_error("Phrase search failed")

    # Semantic mode (default)
    limit = _clamp_int(arguments.get("limit", 10), default=10, max_val=1000)
    event_type = arguments.get("event_type")
    project = arguments.get("project")
    session_id = arguments.get("session_id")
    context_file = arguments.get("context_file")
    context_tags = arguments.get("context_tags")
    filter_tags = arguments.get("filter_tags")
    raw_temporal = arguments.get("temporal_range")
    temporal_range = tuple(raw_temporal) if raw_temporal and len(raw_temporal) == 2 else None
    entity_id = arguments.get("entity_id")
    agent_type = arguments.get("agent_type")

    try:
        from omega.bridge import query

        result = query(
            query_text=query_text,
            limit=limit,
            event_type=event_type,
            project=project,
            session_id=session_id,
            context_file=context_file,
            context_tags=context_tags,
            filter_tags=filter_tags,
            temporal_range=temporal_range,
            entity_id=entity_id,
            agent_type=agent_type,
        )
        return mcp_response(result)
    except Exception as e:
        logger.error("omega_query failed: %s", e)
        return mcp_error("Query failed")


# ============================================================================
# Handler: omega_welcome
# ============================================================================


async def handle_omega_welcome(arguments: dict) -> dict:
    """Get a session welcome briefing with recent relevant memories."""
    session_id = arguments.get("session_id")
    project = arguments.get("project")

    try:
        from omega.bridge import welcome
        from omega import json_compat as json

        briefing = welcome(session_id=session_id, project=project)
        return mcp_response(json.dumps(briefing, indent=2))
    except Exception as e:
        logger.error("omega_welcome failed: %s", e)
        return mcp_error("Welcome briefing failed")


# ============================================================================
# Handler: omega_profile
# ============================================================================


async def handle_omega_profile(arguments: dict) -> dict:
    """Read or update the user profile.

    If 'update' dict is provided, merges those fields and saves.
    Otherwise, returns the current profile.
    Also handles legacy omega_save_profile calls via 'profile' param.
    """
    # Support legacy omega_save_profile param name
    update_data = arguments.get("update") or arguments.get("profile")

    if update_data:
        # Write mode
        try:
            from omega.bridge import get_profile, save_profile

            existing = get_profile()
            existing.pop("preferences_from_memory", None)
            existing.update(update_data)
            success = save_profile(existing)
            if success:
                return mcp_response(f"Profile updated with {len(update_data)} field(s).")
            else:
                return mcp_error("Failed to save profile to disk.")
        except Exception as e:
            logger.error("omega_profile (save) failed: %s", e)
            return mcp_error("Save profile failed")
    else:
        # Read mode
        try:
            from omega.bridge import get_profile
            from omega import json_compat as json

            profile = get_profile()
            if not profile:
                return mcp_response("No profile found. Preferences will build your profile over time.")
            return mcp_response(json.dumps(profile, indent=2))
        except Exception as e:
            logger.error("omega_profile failed: %s", e)
            return mcp_error("Profile failed")


# ============================================================================
# Handler: omega_delete_memory
# ============================================================================


async def handle_omega_delete_memory(arguments: dict) -> dict:
    """Delete a specific memory by its ID."""
    memory_id = arguments.get("memory_id", "").strip()
    if not memory_id:
        return mcp_error("memory_id is required")

    try:
        from omega.bridge import delete_memory

        result = delete_memory(memory_id=memory_id)
        if result.get("success"):
            return mcp_response(f"Deleted memory `{memory_id[:16]}`")
        else:
            return mcp_error(result.get("error", f"Memory {memory_id} not found"))
    except Exception as e:
        logger.error("omega_delete_memory failed: %s", e)
        return mcp_error("Delete failed")


# ============================================================================
# Handler: omega_edit_memory
# ============================================================================


async def handle_omega_edit_memory(arguments: dict) -> dict:
    """Edit the content of a specific memory."""
    memory_id = arguments.get("memory_id", "").strip()
    new_content = arguments.get("new_content", "").strip()

    if not memory_id:
        return mcp_error("memory_id is required")
    if not new_content:
        return mcp_error("new_content is required")

    try:
        from omega.bridge import edit_memory

        result = edit_memory(memory_id=memory_id, new_content=new_content)
        if result.get("success"):
            return mcp_response(f"Updated memory `{memory_id[:16]}`\nNew content: {new_content[:200]}")
        else:
            return mcp_error(result.get("error", f"Memory {memory_id} not found"))
    except Exception as e:
        logger.error("omega_edit_memory failed: %s", e)
        return mcp_error("Edit failed")


# ============================================================================
# Handler: omega_list_preferences
# ============================================================================


async def handle_omega_list_preferences(arguments: dict) -> dict:
    """List all stored user preferences."""
    try:
        from omega.bridge import list_preferences

        prefs = list_preferences()

        if not prefs:
            return mcp_response("No preferences stored yet.")

        lines = [f"## User Preferences ({len(prefs)} total)\n"]
        for pref in prefs:
            content = pref.get("content", "")[:200]
            created = pref.get("created_at", "")[:16]
            pref_id = pref.get("id", "")[:12]
            lines.append(f"- {content}")
            lines.append(f"  _Created: {created} | id: {pref_id}_")
            lines.append("")

        return mcp_response("\n".join(lines))
    except Exception as e:
        logger.error("omega_list_preferences failed: %s", e)
        return mcp_error("List preferences failed")


# ============================================================================
# Handler: omega_health (includes former omega_status stats)
# ============================================================================


async def handle_omega_health(arguments: dict) -> dict:
    """Detailed health check with memory usage, warnings, and recommendations."""
    try:
        from omega.bridge import check_health, status

        warn_mb = _clamp_int(arguments.get("warn_mb", 350), default=350, max_val=10000)
        critical_mb = _clamp_int(arguments.get("critical_mb", 800), default=800, max_val=10000)
        max_nodes = _clamp_int(arguments.get("max_nodes", 10000), default=10000, max_val=100000)
        result = check_health(warn_mb=warn_mb, critical_mb=critical_mb, max_nodes=max_nodes)

        # Append basic stats (formerly omega_status)
        try:
            st = status()
            result += f"**Backend:** {st.get('backend', 'sqlite')}\n"
            result += f"**Store:** {st.get('store_path', '~/.omega')}\n"
            result += f"**Vec enabled:** {st.get('vec_enabled', False)}\n"
        except Exception:
            pass

        return mcp_response(result)
    except Exception as e:
        logger.error("omega_health failed: %s", e)
        return mcp_error("Health check failed")


# ============================================================================
# Handler: omega_backup (merged export + import)
# ============================================================================


async def handle_omega_backup(arguments: dict) -> dict:
    """Export or import memories (backup/restore)."""
    mode = arguments.get("mode", "export").strip()
    filepath = arguments.get("filepath", "").strip()
    if not filepath:
        return mcp_error("filepath is required")

    # Path validation: restrict to ~/.omega/ to prevent sensitive file access
    resolved = Path(filepath).expanduser().resolve()
    safe_dir = _SAFE_EXPORT_DIR.resolve()
    if not str(resolved).startswith(str(safe_dir) + "/") and resolved.parent != safe_dir:
        return mcp_error(f"Path must be under {_SAFE_EXPORT_DIR}")

    if mode == "import":
        if not resolved.exists():
            return mcp_error("File not found")
        clear_existing = arguments.get("clear_existing", True)
        try:
            from omega.bridge import import_memories

            result = import_memories(filepath=str(resolved), clear_existing=clear_existing)
            return mcp_response(result)
        except Exception as e:
            logger.error("omega_backup import failed: %s", e)
            return mcp_error("Import failed (internal error)")
    else:
        try:
            from omega.bridge import export_memories

            result = export_memories(filepath=str(resolved))
            # Warn if encryption is enabled — export is plaintext
            from omega.crypto import is_enabled as crypto_enabled

            if crypto_enabled():
                result["warning"] = (
                    "OMEGA_ENCRYPT is enabled but exports are plaintext. "
                    "The export file contains unencrypted memory content. "
                    "Store it securely or delete after use."
                )
            return mcp_response(result)
        except Exception as e:
            logger.error("omega_backup export failed: %s", e)
            return mcp_error("Export failed (internal error)")


# ============================================================================
# Handler: omega_lessons (merged with omega_cross_project_lessons)
# ============================================================================


async def handle_omega_lessons(arguments: dict) -> dict:
    """Retrieve cross-session or cross-project lessons learned."""
    try:
        cross_project = arguments.get("cross_project", False)
        task = arguments.get("task")
        limit = _clamp_int(arguments.get("limit", 5), default=5, max_val=100)
        agent_type = arguments.get("agent_type")

        if cross_project:
            from omega.bridge import get_cross_project_lessons

            exclude_project = arguments.get("exclude_project")
            exclude_session = arguments.get("exclude_session")
            lessons = get_cross_project_lessons(
                task=task,
                exclude_project=exclude_project,
                exclude_session=exclude_session,
                limit=limit,
                agent_type=agent_type,
            )
            if not lessons:
                return mcp_response("No cross-project lessons found.")

            output = f"# Cross-Project Lessons ({len(lessons)})\n\n"
            for i, lesson in enumerate(lessons, 1):
                proj = lesson.get("source_project", "?")
                xp = " **[CROSS-PROJECT]**" if lesson.get("cross_project") else ""
                output += f"## {i}. {lesson['content'][:120]}\n"
                output += f"*Source: {proj} | Projects seen: {lesson.get('projects_seen', 1)}{xp}*\n"
                output += f"*Accessed: {lesson.get('access_count', 0)} times | Created: {lesson.get('created_at', '?')[:16]}*\n\n"
            return mcp_response(output)
        else:
            from omega.bridge import get_cross_session_lessons

            project_path = arguments.get("project_path")
            lessons = get_cross_session_lessons(
                task=task,
                project_path=project_path,
                limit=limit,
                agent_type=agent_type,
            )
            if not lessons:
                return mcp_response("No cross-session lessons found yet.")

            output = f"# Cross-Session Lessons ({len(lessons)})\n\n"
            for i, lesson in enumerate(lessons, 1):
                verified = " [verified]" if lesson.get("verified") else ""
                access = lesson.get("access_count", 0)
                output += f"## {i}. {lesson.get('content', '')[:200]}{verified}\n"
                output += f"*Access count: {access} | Session: {lesson.get('session_id', 'unknown')[:16]}*\n\n"
            return mcp_response(output)
    except Exception as e:
        logger.error("omega_lessons failed: %s", e)
        return mcp_error("Lessons failed")




# ============================================================================
# Handler: omega_feedback
# ============================================================================


async def handle_omega_feedback(arguments: dict) -> dict:
    """Record feedback on a surfaced memory."""
    memory_id = arguments.get("memory_id", "").strip()
    rating = arguments.get("rating", "").strip()
    reason = arguments.get("reason")

    if not memory_id:
        return mcp_error("memory_id is required")
    if rating not in ("helpful", "unhelpful", "outdated"):
        return mcp_error("rating must be one of: helpful, unhelpful, outdated")

    try:
        from omega.bridge import record_feedback

        result = record_feedback(memory_id=memory_id, rating=rating, reason=reason)
        if "error" in result:
            return mcp_error(result["error"])
        return mcp_response(
            f"Feedback recorded: {rating} for `{memory_id[:16]}`\n"
            f"New score: {result.get('new_score', 0)} "
            f"({result.get('total_signals', 0)} total signals)"
        )
    except Exception as e:
        logger.error("omega_feedback failed: %s", e)
        return mcp_error("Feedback failed")


# ============================================================================
# Handler: omega_clear_session
# ============================================================================


async def handle_omega_clear_session(arguments: dict) -> dict:
    """Clear all memories for a session."""
    session_id = arguments.get("session_id", "").strip()
    if not session_id:
        return mcp_error("session_id is required")

    try:
        from omega.bridge import clear_session

        result = clear_session(session_id=session_id)
        return mcp_response(f"Cleared session `{session_id[:16]}`: {result.get('removed', 0)} memories removed.")
    except Exception as e:
        logger.error("omega_clear_session failed: %s", e)
        return mcp_error("Clear session failed")


# ============================================================================
# Handler: omega_consolidate
# ============================================================================


async def handle_omega_consolidate(arguments: dict) -> dict:
    """Run memory consolidation: prune stale entries, cap summaries, clean edges."""
    prune_days = _clamp_int(arguments.get("prune_days", 30), default=30, max_val=365)
    max_summaries = _clamp_int(arguments.get("max_summaries", 50), default=50, max_val=1000)

    try:
        from omega.bridge import consolidate

        result = consolidate(prune_days=prune_days, max_summaries=max_summaries)
        return mcp_response(result)
    except Exception as e:
        logger.error("omega_consolidate failed: %s", e)
        return mcp_error("Consolidation failed")


# ============================================================================
# Handler: omega_similar
# ============================================================================


async def handle_omega_similar(arguments: dict) -> dict:
    """Find memories similar to a given memory."""
    memory_id = arguments.get("memory_id", "").strip()
    if not memory_id:
        return mcp_error("memory_id is required")

    limit = _clamp_int(arguments.get("limit", 5), default=5, max_val=100)

    try:
        from omega.bridge import find_similar_memories

        result = find_similar_memories(memory_id=memory_id, limit=limit)
        return mcp_response(result)
    except Exception as e:
        logger.error("omega_similar failed: %s", e)
        return mcp_error("Similar search failed")


# ============================================================================
# Handler: omega_timeline
# ============================================================================


async def handle_omega_timeline(arguments: dict) -> dict:
    """Show memory timeline grouped by day."""
    days = _clamp_int(arguments.get("days", 7), default=7, min_val=0, max_val=365)
    limit_per_day = _clamp_int(arguments.get("limit_per_day", 10), default=10, max_val=100)

    try:
        from omega.bridge import timeline

        result = timeline(days=days, limit_per_day=limit_per_day)
        return mcp_response(result)
    except Exception as e:
        logger.error("omega_timeline failed: %s", e)
        return mcp_error("Timeline failed")


# ============================================================================
# Handler: omega_traverse
# ============================================================================


async def handle_omega_traverse(arguments: dict) -> dict:
    """Traverse the memory relationship graph from a starting memory."""
    memory_id = arguments.get("memory_id", "").strip()
    if not memory_id:
        return mcp_error("memory_id is required")

    max_hops = arguments.get("max_hops", 2)
    min_weight = arguments.get("min_weight", 0.0)

    try:
        from omega.bridge import traverse

        result = traverse(
            memory_id=memory_id,
            max_hops=max_hops,
            min_weight=min_weight,
        )
        return mcp_response(result)
    except Exception as e:
        logger.error("omega_traverse failed: %s", e)
        return mcp_error("Traverse failed")


# ============================================================================
# Handler: omega_compact
# ============================================================================


async def handle_omega_compact(arguments: dict) -> dict:
    """Compact related memories into consolidated knowledge nodes."""
    event_type = arguments.get("event_type", "lesson_learned")
    similarity_threshold = arguments.get("similarity_threshold", 0.6)
    min_cluster_size = _clamp_int(arguments.get("min_cluster_size", 3), default=3, min_val=2, max_val=100)
    dry_run = arguments.get("dry_run", False)

    try:
        from omega.bridge import compact

        result = compact(
            event_type=event_type,
            similarity_threshold=similarity_threshold,
            min_cluster_size=min_cluster_size,
            dry_run=dry_run,
        )
        return mcp_response(result)
    except Exception as e:
        logger.error("omega_compact failed: %s", e)
        return mcp_error("Compact failed")




# ============================================================================
# Handler: omega_type_stats
# ============================================================================


async def handle_omega_type_stats(arguments: dict) -> dict:
    """Get memory counts grouped by event type."""
    try:
        from omega.bridge import type_stats

        stats = type_stats()
        if not stats:
            return mcp_response("No memories stored yet.")

        total = sum(stats.values())
        lines = [f"# Memory Type Stats ({total} total)\n"]
        for etype, count in sorted(stats.items(), key=lambda x: x[1], reverse=True):
            pct = (count / total * 100) if total > 0 else 0
            lines.append(f"- **{etype}**: {count} ({pct:.1f}%)")
        return mcp_response("\n".join(lines))
    except Exception as e:
        logger.error("omega_type_stats failed: %s", e)
        return mcp_error("Type stats failed")


# ============================================================================
# Handler: omega_session_stats
# ============================================================================


async def handle_omega_session_stats(arguments: dict) -> dict:
    """Get memory counts grouped by session ID."""
    try:
        from omega.bridge import session_stats

        stats = session_stats()
        if not stats:
            return mcp_response("No session data found.")

        # Sort by count descending, show top 20
        sorted_sessions = sorted(stats.items(), key=lambda x: x[1], reverse=True)[:20]
        total = sum(stats.values())
        lines = [f"# Session Stats (top {len(sorted_sessions)} of {len(stats)} sessions, {total} total memories)\n"]
        for sid, count in sorted_sessions:
            truncated = sid[:16] + "..." if len(sid) > 16 else sid
            lines.append(f"- `{truncated}`: {count} memories")
        return mcp_response("\n".join(lines))
    except Exception as e:
        logger.error("omega_session_stats failed: %s", e)
        return mcp_error("Session stats failed")


# ============================================================================
# Handler: omega_weekly_digest
# ============================================================================


async def handle_omega_weekly_digest(arguments: dict) -> dict:
    """Generate a weekly knowledge digest with stats, trends, and highlights."""
    try:
        from omega.bridge import get_weekly_digest

        days = arguments.get("days", 7)
        digest = get_weekly_digest(days=days)

        lines = [f"# Your Week in Review ({digest['period_days']}d)\n"]

        # Summary line
        lines.append(
            f"**{digest['period_new']} new memories** across "
            f"{digest['session_count']} sessions "
            f"({digest['total_memories']} total)"
        )

        # Growth
        if digest["prev_period_count"] > 0:
            direction = "up" if digest["growth_pct"] > 0 else "down"
            lines.append(
                f"**Growth:** {direction} {abs(digest['growth_pct'])}% vs previous {days}d "
                f"({digest['prev_period_count']} -> {digest['period_new']})"
            )

        # Type breakdown
        if digest["type_breakdown"]:
            lines.append("\n**Breakdown:**")
            for etype, count in sorted(digest["type_breakdown"].items(), key=lambda x: x[1], reverse=True):
                if count > 0 and etype != "session_summary":
                    lines.append(f"  - {etype}: {count}")

        # Top topics
        if digest["top_topics"]:
            lines.append(f"\n**Top topics:** {', '.join(digest['top_topics'][:6])}")

        return mcp_response("\n".join(lines))
    except Exception as e:
        logger.error("omega_weekly_digest failed: %s", e)
        return mcp_error("Weekly digest failed")


# ============================================================================
# Handler: omega_checkpoint
# ============================================================================


async def handle_omega_checkpoint(arguments: dict) -> dict:
    """Save a task checkpoint for session continuity."""
    task_title = arguments.get("task_title", "").strip()
    progress = arguments.get("progress", "").strip()
    if not task_title or not progress:
        return mcp_error("task_title and progress are required")

    # Build structured checkpoint content
    checkpoint = {
        "version": 1,
        "task_title": task_title,
        "plan": arguments.get("plan", ""),
        "progress": progress,
        "files_touched": arguments.get("files_touched", {}),
        "decisions": arguments.get("decisions", []),
        "key_context": arguments.get("key_context", ""),
        "next_steps": arguments.get("next_steps", ""),
    }

    # Format as searchable text content
    content_lines = [f"## Checkpoint: {task_title}"]
    if checkpoint["plan"]:
        content_lines.append(f"\n### Plan\n{checkpoint['plan']}")
    content_lines.append(f"\n### Progress\n{checkpoint['progress']}")
    if checkpoint["files_touched"]:
        content_lines.append("\n### Files Changed")
        for fp, summary in checkpoint["files_touched"].items():
            content_lines.append(f"- `{fp}`: {summary}")
    if checkpoint["decisions"]:
        content_lines.append("\n### Decisions")
        for d in checkpoint["decisions"]:
            content_lines.append(f"- {d}")
    if checkpoint["key_context"]:
        content_lines.append(f"\n### Key Context\n{checkpoint['key_context']}")
    if checkpoint["next_steps"]:
        content_lines.append(f"\n### Next Steps\n{checkpoint['next_steps']}")

    content = "\n".join(content_lines)

    # Determine checkpoint number for this task
    session_id = arguments.get("session_id")
    project = arguments.get("project")
    checkpoint_num = 1
    try:
        from omega.bridge import query_structured

        existing = query_structured(
            query_text=f"checkpoint {task_title}",
            limit=10,
            event_type="checkpoint",
        )
        if project:
            existing = [e for e in existing if (e.get("metadata") or {}).get("project") == project]
        checkpoint_num = len(existing) + 1
    except Exception:
        pass

    metadata = {
        "checkpoint_number": checkpoint_num,
        "checkpoint_data": checkpoint,
    }

    try:
        from omega.bridge import auto_capture

        result = auto_capture(
            content=content,
            event_type="checkpoint",
            metadata=metadata,
            session_id=session_id,
            project=project,
        )
        return mcp_response(f"{result}\n\nCheckpoint #{checkpoint_num} saved for: {task_title}")
    except Exception as e:
        logger.error("omega_checkpoint failed: %s", e)
        return mcp_error(f"Checkpoint failed: {e}")


# ============================================================================
# Handler: omega_resume_task
# ============================================================================


async def handle_omega_resume_task(arguments: dict) -> dict:
    """Resume a checkpointed task with full context."""
    task_title = arguments.get("task_title", "").strip()
    project = arguments.get("project")
    verbosity = arguments.get("verbosity", "full")
    limit = _clamp_int(arguments.get("limit"), 1, 1, 5)

    # Build search query
    query_text = f"checkpoint {task_title}" if task_title else "checkpoint"

    try:
        from omega.bridge import query_structured

        results = query_structured(
            query_text=query_text,
            limit=limit * 3,  # Over-fetch for filtering
            event_type="checkpoint",
        )

        if not results:
            return mcp_response("No checkpoints found. Start fresh or provide a different task title.")

        # Post-filter by project if specified (metadata match, not query dilution)
        if project:
            filtered = [r for r in results if (r.get("metadata") or {}).get("project") == project]
            if filtered:
                results = filtered

        # Take the most recent checkpoints (by created_at)
        results = sorted(results, key=lambda r: r.get("created_at", ""), reverse=True)[:limit]

        lines = [f"# Task Resume — {len(results)} checkpoint(s) found\n"]

        for r in results:
            meta = r.get("metadata", {})
            checkpoint_data = meta.get("checkpoint_data", {})
            cp_num = meta.get("checkpoint_number", "?")
            created = r.get("created_at", "unknown")[:16]

            if verbosity == "minimal":
                next_steps = checkpoint_data.get("next_steps", "No next steps recorded")
                lines.append(f"## Checkpoint #{cp_num} ({created})")
                lines.append(f"**Task**: {checkpoint_data.get('task_title', 'Unknown')}")
                lines.append(f"**Next Steps**: {next_steps}\n")
            elif verbosity == "summary":
                lines.append(f"## Checkpoint #{cp_num} ({created})")
                lines.append(f"**Task**: {checkpoint_data.get('task_title', 'Unknown')}")
                if checkpoint_data.get("plan"):
                    lines.append(f"**Plan**: {checkpoint_data['plan']}")
                lines.append(f"**Progress**: {checkpoint_data.get('progress', 'Unknown')}")
                lines.append(f"**Next Steps**: {checkpoint_data.get('next_steps', 'None')}\n")
            else:  # full
                lines.append(r.get("content", "No content"))
                if checkpoint_data.get("files_touched") and "Files Changed" not in r.get("content", ""):
                    lines.append("\n### Files Changed")
                    for fp, summary in checkpoint_data["files_touched"].items():
                        lines.append(f"- `{fp}`: {summary}")
                lines.append("")

        return mcp_response("\n".join(lines))
    except Exception as e:
        logger.error("omega_resume_task failed: %s", e)
        return mcp_error(f"Resume failed: {e}")


# ============================================================================
# Handler: omega_remind
# ============================================================================


async def handle_omega_remind(arguments: dict) -> dict:
    """Create a time-based reminder."""
    text = arguments.get("text", "").strip()
    duration = arguments.get("duration", "").strip()
    if not text:
        return mcp_error("text is required")
    if not duration:
        return mcp_error("duration is required (e.g. '1h', '30m', '2d')")

    context = arguments.get("context")
    session_id = arguments.get("session_id")
    project = arguments.get("project")

    try:
        from omega.bridge import create_reminder

        result = create_reminder(
            text=text,
            duration=duration,
            context=context,
            session_id=session_id,
            project=project,
        )
        lines = [
            f"Reminder set: {result['text']}",
            f"Due at: {result['remind_at_local']}",
            f"ID: {result['reminder_id']}",
        ]
        return mcp_response("\n".join(lines))
    except ValueError as e:
        return mcp_error(str(e))
    except Exception as e:
        logger.error("omega_remind failed: %s", e)
        return mcp_error(f"Failed to create reminder: {e}")


# ============================================================================
# Handler: omega_remind_list
# ============================================================================


async def handle_omega_remind_list(arguments: dict) -> dict:
    """List reminders with status and due times."""
    status = arguments.get("status")

    try:
        from omega.bridge import list_reminders

        include_dismissed = status in ("dismissed", "all")
        reminders = list_reminders(status=status, include_dismissed=include_dismissed)

        if not reminders:
            return mcp_response("No reminders found.")

        lines = [f"**Reminders** ({len(reminders)} found)\n"]
        status_icons = {"pending": "⏳", "fired": "🔔", "dismissed": "✓"}
        for r in reminders:
            icon = status_icons.get(r["status"], "?")
            overdue = " **[OVERDUE]**" if r.get("is_overdue") else ""
            lines.append(f"- {icon} {r['text']}{overdue}")
            lines.append(f"  Due: {r['remind_at_local']} | Status: {r['status']} | Time: {r['time_until']}")
            if r.get("context"):
                lines.append(f"  Context: {r['context'][:120]}")
            lines.append(f"  ID: {r['id']}")

        return mcp_response("\n".join(lines))
    except Exception as e:
        logger.error("omega_remind_list failed: %s", e)
        return mcp_error(f"Failed to list reminders: {e}")


# ============================================================================
# Handler: omega_remind_dismiss
# ============================================================================


async def handle_omega_remind_dismiss(arguments: dict) -> dict:
    """Dismiss a reminder by ID."""
    reminder_id = arguments.get("reminder_id", "").strip()
    if not reminder_id:
        return mcp_error("reminder_id is required")

    try:
        from omega.bridge import dismiss_reminder

        result = dismiss_reminder(reminder_id)
        if result.get("success"):
            return mcp_response(f"Dismissed reminder: {result.get('text', reminder_id)}")
        return mcp_error(result.get("error", "Failed to dismiss reminder"))
    except Exception as e:
        logger.error("omega_remind_dismiss failed: %s", e)
        return mcp_error(f"Failed to dismiss reminder: {e}")


# ============================================================================
# Handler: omega_protocol
# ============================================================================


async def handle_omega_protocol(arguments: dict) -> dict:
    """Serve the coordination playbook dynamically based on context."""
    section = arguments.get("section")
    project = arguments.get("project")

    # Detect peer count for auto-mode selection
    peer_count = 0
    try:
        from omega.coordination import get_manager

        mgr = get_manager()
        sessions = mgr.list_sessions(auto_clean=True)
        peer_count = max(0, len(sessions) - 1)
    except Exception:
        pass  # Solo mode if coordination unavailable

    try:
        from omega.protocol import get_protocol

        result = get_protocol(
            section=section,
            project=project,
            include_lessons=True,
            peer_count=peer_count,
        )
        return mcp_response(result)
    except Exception as e:
        logger.error("omega_protocol failed: %s", e, exc_info=True)
        return mcp_error(f"Protocol failed: {e}")


# ============================================================================
# Handler Registry
# ============================================================================

HANDLERS: Dict[str, Any] = {
    "omega_remember": lambda args: handle_omega_store(
        {**args, "event_type": args.get("event_type", "user_preference")}
    ),  # Alias — backward compat (defaults to user_preference like old remember)
    "omega_store": handle_omega_store,
    "omega_query": handle_omega_query,
    "omega_welcome": handle_omega_welcome,
    "omega_profile": handle_omega_profile,
    "omega_delete_memory": handle_omega_delete_memory,
    "omega_edit_memory": handle_omega_edit_memory,
    "omega_list_preferences": handle_omega_list_preferences,
    "omega_health": handle_omega_health,
    "omega_backup": handle_omega_backup,
    "omega_lessons": handle_omega_lessons,
    "omega_save_profile": handle_omega_profile,  # Alias — backward compat
    "omega_feedback": handle_omega_feedback,
    "omega_clear_session": handle_omega_clear_session,
    "omega_similar": handle_omega_similar,
    "omega_timeline": handle_omega_timeline,
    "omega_consolidate": handle_omega_consolidate,
    "omega_traverse": handle_omega_traverse,
    "omega_compact": handle_omega_compact,
    "omega_phrase_search": lambda args: handle_omega_query(
        {**args, "query": args.get("phrase", args.get("query", "")), "mode": "phrase"}
    ),  # Alias — backward compat
    "omega_checkpoint": handle_omega_checkpoint,
    "omega_resume_task": handle_omega_resume_task,
    "omega_type_stats": handle_omega_type_stats,
    "omega_session_stats": handle_omega_session_stats,
    "omega_weekly_digest": handle_omega_weekly_digest,
    "omega_remind": handle_omega_remind,
    "omega_remind_list": handle_omega_remind_list,
    "omega_remind_dismiss": handle_omega_remind_dismiss,
    "omega_protocol": handle_omega_protocol,
}
