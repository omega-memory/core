"""OMEGA Hook Server — Unix Domain Socket daemon for fast hook dispatch.

Runs inside the MCP server process, reusing warm bridge/coordination singletons.
Hooks connect via ~/.omega/hook.sock, send a JSON request, and get a JSON response.
This eliminates ~750ms of cold-start overhead per hook invocation.
"""

import asyncio
import hashlib
import json
import logging
import os
import re
import subprocess
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger("omega.hook_server")

SOCK_PATH = Path.home() / ".omega" / "hook.sock"

# Debounce state (in-memory, reset on server restart)
_last_surface: dict[str, float] = {}  # file_path -> timestamp (capped at 500)
_MAX_SURFACE_ENTRIES = 500
_last_peer_dir_check: dict[str, float] = {}  # dir_path -> timestamp
PEER_DIR_CHECK_DEBOUNCE_S = 300.0  # 5 minutes
_last_coord_query: dict[str, float] = {}  # session_id -> monotonic timestamp
COORD_QUERY_DEBOUNCE_S = 120.0  # 2 minutes
_last_reminder_check: float = 0.0
REMINDER_CHECK_DEBOUNCE_S = 300.0  # 5 minutes
SURFACE_DEBOUNCE_S = 5.0

# Urgent message queue — push notifications from send_message to recipient's next heartbeat
_pending_urgent: dict[str, list[dict]] = {}  # session_id -> urgent message summaries
_MAX_URGENT_PER_SESSION = 10


# Error dedup state (mirrors standalone surface_memories.py)
_error_hashes: set = set()  # capped at 200 per session
_MAX_ERROR_HASHES = 200
_error_counts: dict[str, int] = {}  # session_id -> error count
_MAX_ERRORS_PER_SESSION = 5


# ---------------------------------------------------------------------------
# Agent nicknames — deterministic human-readable names from session IDs
# ---------------------------------------------------------------------------

_AGENT_NAMES = [
    "Alder",
    "Aspen",
    "Birch",
    "Briar",
    "Brook",
    "Cedar",
    "Cliff",
    "Cloud",
    "Coral",
    "Crane",
    "Creek",
    "Dune",
    "Elm",
    "Ember",
    "Fern",
    "Finch",
    "Flint",
    "Frost",
    "Glen",
    "Grove",
    "Hawk",
    "Hazel",
    "Heath",
    "Heron",
    "Holly",
    "Iris",
    "Ivy",
    "Jade",
    "Jay",
    "Juniper",
    "Lake",
    "Lark",
    "Laurel",
    "Leaf",
    "Lily",
    "Maple",
    "Marsh",
    "Meadow",
    "Moss",
    "Oak",
    "Olive",
    "Opal",
    "Orca",
    "Osprey",
    "Pearl",
    "Pebble",
    "Pine",
    "Rain",
    "Raven",
    "Reed",
    "Ridge",
    "Robin",
    "Sage",
    "Shore",
    "Sky",
    "Slate",
    "Stone",
    "Storm",
    "Swift",
    "Thorn",
    "Tide",
    "Vale",
    "Willow",
    "Wren",
]


def _agent_nickname(session_id: str) -> str:
    """Generate a deterministic, memorable nickname from a session ID.

    Returns format: "Nickname (abcd1234)" — e.g. "Cedar (a3f2b1c8)".
    """
    if not session_id:
        return "unknown"
    idx = int(hashlib.md5(session_id.encode()).hexdigest()[:8], 16) % len(_AGENT_NAMES)
    return f"{_AGENT_NAMES[idx]} ({session_id[:8]})"


def notify_session(target_session_id: str, msg_summary: dict) -> None:
    """Queue an urgent message notification for a target session.

    Called from coord_handlers after a successful send_message. The recipient's
    next heartbeat (even if debounced) will drain and surface these.
    """
    try:
        if not target_session_id or not msg_summary:
            return
        queue = _pending_urgent.setdefault(target_session_id, [])
        queue.append(msg_summary)
        # Cap at max, keeping newest
        if len(queue) > _MAX_URGENT_PER_SESSION:
            _pending_urgent[target_session_id] = queue[-_MAX_URGENT_PER_SESSION:]
    except Exception:
        pass  # Fail-open


def _drain_urgent_queue(session_id: str) -> str:
    """Pop and format all pending urgent messages for a session.

    Returns formatted [INBOX] lines or empty string.
    """
    queue = _pending_urgent.pop(session_id, None)
    if not queue:
        return ""
    try:
        previews = []
        for msg in queue:
            from_name = _agent_nickname(msg.get("from_session") or "unknown")
            subj = (msg.get("subject") or "")[:60]
            mtype = msg.get("msg_type", "inform")
            previews.append(f'{from_name} [{mtype}]: "{subj}"')
        return "[INBOX] " + " | ".join(previews) + " — use omega_inbox for details"
    except Exception:
        return ""


def _secure_append(log_path: Path, data: str):
    """Append to a file with secure permissions (0o600)."""
    log_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd = os.open(str(log_path), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        os.write(fd, data.encode("utf-8"))
    finally:
        os.close(fd)


def _log_hook_error(hook_name: str, error: Exception):
    """Log hook errors to ~/.omega/hooks.log."""
    try:
        log_path = Path.home() / ".omega" / "hooks.log"
        timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        tb = traceback.format_exc()
        _secure_append(log_path, f"[{timestamp}] hook_server/{hook_name}: {error}\n{tb}\n")
    except Exception:
        pass


def _log_timing(hook_name: str, elapsed_ms: float):
    """Log hook timing to ~/.omega/hooks.log."""
    try:
        log_path = Path.home() / ".omega" / "hooks.log"
        timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        _secure_append(log_path, f"[{timestamp}] hook_server/{hook_name}: OK ({elapsed_ms:.0f}ms)\n")
    except Exception:
        pass


def _auto_cloud_sync(session_id: str):
    """Fire-and-forget cloud sync on session stop. Fully fail-open."""
    try:
        secrets_path = Path.home() / ".omega" / "secrets.json"
        if not secrets_path.exists():
            return  # Cloud not configured — fast bail

        import threading

        def _sync():
            t0 = time.monotonic()
            try:
                from omega.cloud.sync import get_sync

                get_sync().sync_all()
                # Write push marker for status tracking
                push_marker = Path.home() / ".omega" / "last-cloud-push"
                push_marker.parent.mkdir(parents=True, exist_ok=True)
                push_marker.write_text(datetime.now(timezone.utc).isoformat())
                _log_timing("auto_cloud_sync", (time.monotonic() - t0) * 1000)
            except Exception as e:
                _log_hook_error("auto_cloud_sync", e)

        t = threading.Thread(target=_sync, daemon=True, name="omega-cloud-sync")
        t.start()
    except Exception:
        pass  # Never propagate


def _resolve_entity(project: str) -> "Optional[str]":
    """Resolve project→entity_id. Fail-open: returns None on any error.

    Delegates to resolve_project_entity() which reads config-file mappings.
    Returns None when no mappings exist or no match found.
    """
    try:
        from omega.entity.engine import resolve_project_entity

        return resolve_project_entity(project)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _relative_time_from_iso(iso_str: str) -> str:
    """Convert an ISO timestamp to a human-readable relative time like '2d ago'."""
    try:
        dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        secs = (datetime.now(timezone.utc) - dt).total_seconds()
        if secs < 0:
            return "just now"
        if secs < 60:
            return f"{int(secs)}s ago"
        if secs < 3600:
            return f"{int(secs / 60)}m ago"
        if secs < 86400:
            return f"{int(secs / 3600)}h ago"
        return f"{int(secs / 86400)}d ago"
    except Exception:
        return ""


def _check_milestone(name: str) -> bool:
    """Return True if milestone not yet achieved (first time). Creates marker."""
    marker = Path.home() / ".omega" / "milestones" / name
    if marker.exists():
        return False
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.touch()
    return True


def _should_run_periodic(marker_name: str, interval_seconds: int) -> bool:
    """Check if a periodic task should run based on its marker file age.

    Returns True if the marker is missing or older than interval_seconds.
    Updates the marker timestamp on caller's behalf (caller writes after success).
    """
    marker = Path.home() / ".omega" / marker_name
    if not marker.exists():
        return True
    try:
        last_ts = marker.read_text().strip()
        last = datetime.fromisoformat(last_ts)
        if last.tzinfo is None:
            last = last.replace(tzinfo=timezone.utc)
        age_seconds = (datetime.now(timezone.utc) - last).total_seconds()
        return age_seconds >= interval_seconds
    except Exception:
        return True


def _update_marker(marker_name: str) -> None:
    """Write current UTC timestamp to a marker file."""
    marker = Path.home() / ".omega" / marker_name
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(datetime.now(timezone.utc).isoformat())


def _parse_tool_input(payload: dict) -> dict:
    """Parse tool_input from a hook payload into a dict. Returns {} on failure."""
    raw = payload.get("tool_input", "{}")
    try:
        return json.loads(raw) if isinstance(raw, str) else (raw or {})
    except (json.JSONDecodeError, TypeError):
        return {}


def _get_file_path_from_input(input_data: dict) -> str:
    """Extract file_path or notebook_path from parsed tool input."""
    return input_data.get("file_path", input_data.get("notebook_path", ""))


def _debounce_check(cache: dict, key, debounce_s: float, max_entries: int) -> bool:
    """Check if a key has been seen within debounce_s seconds.

    Returns True if the action should proceed (not debounced).
    Updates the cache timestamp and evicts the oldest entry if over max_entries.
    """
    now = time.monotonic()
    if key in cache and now - cache[key] < debounce_s:
        return False
    cache[key] = now
    if len(cache) > max_entries:
        oldest = min(cache, key=cache.get)
        del cache[oldest]
    return True


def _format_age(dt_str: str) -> str:
    """Format a datetime ISO string as a human-readable relative age.

    Returns e.g. "5m ago", "2h15m ago", or "" on failure.
    """
    try:
        dt = datetime.fromisoformat(dt_str)
        if dt.tzinfo is not None:
            dt = dt.replace(tzinfo=None)
        delta = datetime.now(timezone.utc).replace(tzinfo=None) - dt
        mins = int(delta.total_seconds() / 60)
        if mins < 60:
            return f"{mins}m ago"
        return f"{mins // 60}h{mins % 60}m ago"
    except Exception:
        return ""


def _append_output(existing: str, new_line: str) -> str:
    """Append a line to output with newline separator."""
    return (existing + "\n" + new_line) if existing else new_line


# ---------------------------------------------------------------------------
# Handler functions — replicate hook script logic using warm singletons
# ---------------------------------------------------------------------------


def handle_session_start(payload: dict) -> dict:
    """Welcome briefing + auto-consolidation check."""
    session_id = payload.get("session_id", "")
    project = payload.get("project", "")

    # Auto-consolidation check (max once per 7 days)
    try:
        if _should_run_periodic("last-consolidate", 7 * 86400):
            from omega.bridge import consolidate

            consolidate(prune_days=30, max_summaries=50)
            _update_marker("last-consolidate")
    except Exception as e:
        _log_hook_error("auto_consolidate", e)

    # Auto-compaction check (max once per 14 days)
    try:
        if _should_run_periodic("last-compact", 14 * 86400):
            from omega.bridge import compact

            compact(event_type="lesson_learned", similarity_threshold=0.60, min_cluster_size=3)
            _update_marker("last-compact")
    except Exception as e:
        _log_hook_error("auto_compact", e)

    # Auto-backup check (max once per 7 days)
    try:
        if _should_run_periodic("last-backup", 7 * 86400):
            backup_dir = Path.home() / ".omega" / "backups"
            backup_dir.mkdir(parents=True, exist_ok=True)
            dest = backup_dir / f"omega-{datetime.now(timezone.utc).strftime('%Y-%m-%d')}.json"
            from omega.bridge import export_memories

            export_memories(filepath=str(dest))
            # Rotate: keep only last 4
            backups = sorted(backup_dir.glob("omega-*.json"), key=lambda p: p.name, reverse=True)
            for old in backups[4:]:
                old.unlink()
            _update_marker("last-backup")
    except Exception as e:
        _log_hook_error("auto_backup", e)

    # Auto-doctor check (max once per 7 days)
    doctor_summary = ""
    try:
        if _should_run_periodic("last-doctor", 7 * 86400):
            from omega.bridge import status as omega_status

            s = omega_status()
            issues = []
            if s.get("node_count", 0) == 0:
                issues.append("0 memories")
            if not s.get("vec_enabled"):
                issues.append("vec disabled")
            # FTS5 integrity check
            try:
                import sqlite3 as _sqlite3

                db_path = Path.home() / ".omega" / "omega.db"
                if db_path.exists():
                    _conn = _sqlite3.connect(str(db_path))
                    _conn.execute("INSERT INTO memories_fts(memories_fts) VALUES('integrity-check')")
                    _conn.close()
            except Exception:
                issues.append("FTS5 integrity issue")
            doctor_summary = f"doctor: {len(issues)} issue(s)" if issues else "doctor: healthy"
            _update_marker("last-doctor")
    except Exception as e:
        _log_hook_error("auto_doctor", e)

    # Auto-scan documents folder (max once per hour)
    doc_scan_summary = ""
    try:
        docs_dir = Path.home() / ".omega" / "documents"
        if _should_run_periodic("last-doc-scan", 3600) and docs_dir.exists() and any(docs_dir.iterdir()):
            from omega.knowledge.engine import scan_directory

            result = scan_directory()
            # Only show summary if something was ingested
            if "ingested" in result.lower() and "0 ingested" not in result:
                doc_scan_summary = result
            _update_marker("last-doc-scan")
    except Exception as e:
        _log_hook_error("auto_doc_scan", e)

    # Auto-pull from cloud (max once per day — pull is fast, just checks hashes)
    cloud_pull_summary = ""
    try:
        secrets_path = Path.home() / ".omega" / "secrets.json"
        if secrets_path.exists() and _should_run_periodic("last-cloud-pull", 86400):
            from omega.cloud.sync import get_sync

            result = get_sync().pull_all()
            mem_pulled = result.get("memories", {}).get("pulled", 0)
            doc_pulled = result.get("documents", {}).get("pulled", 0)
            total_pulled = mem_pulled + doc_pulled
            if total_pulled > 0:
                parts = []
                if mem_pulled:
                    parts.append(f"{mem_pulled} memories")
                if doc_pulled:
                    parts.append(f"{doc_pulled} documents")
                cloud_pull_summary = f"cloud: pulled {', '.join(parts)}"
            _update_marker("last-cloud-pull")
    except Exception as e:
        _log_hook_error("auto_cloud_pull", e)

    # Clean up stale surfacing counter files (both .surfaced and .surfaced.json)
    try:
        omega_dir = Path.home() / ".omega"
        cutoff = time.time() - 86400
        for pattern in ("session-*.surfaced", "session-*.surfaced.json"):
            for f in omega_dir.glob(pattern):
                if f.stat().st_mtime < cutoff:
                    f.unlink()
    except Exception:
        pass

    # Gather session context for briefing
    try:
        from omega.bridge import get_session_context

        ctx = get_session_context(project=project, exclude_session=session_id)
    except Exception as e:
        _log_hook_error("session_start", e)
        return {"output": f"OMEGA welcome failed: {e}", "error": str(e)}

    memory_count = ctx.get("memory_count", 0)
    health_status = ctx.get("health_status", "ok")
    last_capture = ctx.get("last_capture_ago", "unknown")
    context_items = ctx.get("context_items", [])

    # Detect project name and git branch/status
    project_name = Path(project).name if project else "unknown"
    git_branch = _get_current_branch(project or ".") or "unknown"
    git_status_str = "unknown"
    try:
        status_result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=project or ".",
        )
        if status_result.returncode == 0:
            changed = len([l for l in status_result.stdout.strip().split("\n") if l.strip()])
            git_status_str = "Clean" if changed == 0 else f"{changed} unstaged changes"
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass

    # --- Section 1: Header (always) ---
    first_word = "Welcome back!" if memory_count > 0 else "Welcome!"
    lines = [
        f"## {first_word} OMEGA ready — {memory_count} memories | Project: {project_name} | Branch: {git_branch} | {git_status_str}"
    ]

    # First-time user "Aha" moment — guided welcome for new users
    if memory_count == 0:
        lines.append("")
        lines.append("OMEGA captures decisions, lessons, and errors automatically as you work.")
        lines.append("Next session, it surfaces relevant context when you edit the same files.")
        lines.append("")
        lines.append("**Quick start:**")
        lines.append('- Say "remember that we always use TypeScript strict mode" to store a preference')
        lines.append("- Make a decision and OMEGA captures it automatically")
        lines.append("- Encounter an error, and OMEGA stores the pattern for future recall")
        lines.append("")
        lines.append("After this session ends, you'll see exactly what was captured.")
    elif memory_count <= 10:
        lines.append(f"  OMEGA has {memory_count} memories from your first sessions. These will surface when you edit related files.")
        try:
            from omega.bridge import type_stats as _ts_first

            first_stats = _ts_first()
            stat_parts = []
            for k, v in sorted(first_stats.items(), key=lambda x: x[1], reverse=True):
                if v > 0 and k != "session_summary":
                    stat_parts.append(f"{v} {k.replace('_', ' ')}")
            if stat_parts:
                lines.append(f"  Captured so far: {', '.join(stat_parts[:4])}")
        except Exception:
            pass

    # --- Section 2: Health line (always) ---
    health_line = f"**Health:** {health_status} | **Last capture:** {last_capture}"
    lines.append(health_line)

    # --- Section 2a: Alerts for degraded subsystems ---
    # Embedding model warning → [!] alert
    try:
        from omega.graphs import get_active_backend

        if get_active_backend() is None:
            lines.append("[!] Embedding model unavailable — semantic search degraded (hash fallback)")
    except Exception:
        pass

    # Router degradation → [!] alert (only when providers degraded)
    try:
        from omega.router.engine import OmegaRouter

        router = OmegaRouter()
        provider_status = router.get_provider_status()
        available = sum(1 for s in provider_status.values() if s == "available")
        total = len(provider_status)
        if 0 < available < total:
            lines.append(f"[!] Router: {available}/{total} providers active — some routing degraded")
        elif available == 0 and total > 0:
            lines.append("[!] Router: 0 providers active — routing unavailable")
    except ImportError:
        pass  # Router is optional
    except Exception as e:
        _log_hook_error("router_status_welcome", e)

    # Doctor issues → [!] alert
    if doctor_summary and "issue" in doctor_summary:
        lines.append(f"[!] {doctor_summary}")

    # Document scan results (only if new files were ingested)
    if doc_scan_summary:
        lines.append(f"[DOCS] {doc_scan_summary}")

    # Cloud pull results (only if new data was pulled)
    if cloud_pull_summary:
        lines.append(f"[CLOUD] {cloud_pull_summary}")

    # --- Section 3: [REMINDER] due reminders with smart enrichment ---
    try:
        from omega.bridge import get_due_reminders

        due_reminders = get_due_reminders(mark_fired=True)
        if due_reminders:
            lines.append("")
            for r in due_reminders[:5]:  # Cap at 5
                overdue_label = " [OVERDUE]" if r.get("is_overdue") else ""
                lines.append(f"[REMINDER]{overdue_label} {r['text']}")
                if r.get("context"):
                    lines.append(f"  Context: {r['context'][:120]}")

                # Smart enrichment: find related memories for this reminder
                try:
                    from omega.bridge import query_structured as _qs_enrich

                    reminder_text = r.get("text", "")
                    if len(reminder_text) > 10:
                        related = _qs_enrich(
                            query_text=reminder_text[:200],
                            limit=2,
                            event_type="decision",
                        )
                        related_lessons = _qs_enrich(
                            query_text=reminder_text[:200],
                            limit=1,
                            event_type="lesson_learned",
                        )
                        enrichments = []
                        for m in (related or []) + (related_lessons or []):
                            if m.get("relevance", 0) >= 0.30:
                                preview = m.get("content", "")[:80].replace("\n", " ").strip()
                                etype = m.get("event_type", "")
                                if preview:
                                    enrichments.append(f"{etype}: {preview}")
                        if enrichments:
                            lines.append(f"  Related: {enrichments[0]}")
                except Exception:
                    pass

                lines.append(f"  ID: {r['id'][:12]} — dismiss with omega_remind_dismiss")
    except Exception as e:
        _log_hook_error("reminder_check", e)

    # --- Section 5: [CONTEXT] (if items exist) ---
    if context_items:
        lines.append("")
        lines.append("[CONTEXT]")
        for item in context_items:
            lines.append(f"  {item['tag']}: {item['text']}")

    # --- Section 7: [ACTION] maintenance nudges ---
    try:
        from omega.bridge import type_stats as _ts

        stats = _ts()
        lesson_count = stats.get("lesson_learned", 0)
        if lesson_count >= 40:
            lines.append(
                f"\n[ACTION] {lesson_count} lessons with potential duplicates — run omega_compact if asked about maintenance"
            )
    except Exception:
        pass

    # --- Section 7a: Expanded welcome nudges ---
    nudges: list[str] = []

    # Nudge: overdue backup
    try:
        backup_marker = Path.home() / ".omega" / "last-backup"
        if backup_marker.exists():
            last_ts = backup_marker.read_text().strip()
            last = datetime.fromisoformat(last_ts)
            if last.tzinfo is None:
                last = last.replace(tzinfo=timezone.utc)
            age_days = (datetime.now(timezone.utc) - last).days
            if age_days >= 14:
                nudges.append(f"Last backup: {age_days}d ago — consider running omega_backup")
        elif memory_count > 50:
            nudges.append("No backup found — consider running omega_backup")
    except Exception:
        pass

    # Nudge: due/overdue reminders count
    try:
        from omega.bridge import list_reminders as _lr

        pending = _lr(status="pending")
        due_count = sum(1 for r in pending if r.get("is_due"))
        upcoming_today = 0
        for r in pending:
            if not r.get("is_due"):
                try:
                    remind_at = datetime.fromisoformat(r["remind_at"])
                    if remind_at.tzinfo is None:
                        remind_at = remind_at.replace(tzinfo=timezone.utc)
                    if (remind_at - datetime.now(timezone.utc)).total_seconds() < 86400:
                        upcoming_today += 1
                except Exception:
                    pass
        if due_count > 0:
            nudges.append(f"{due_count} reminder{'s' if due_count != 1 else ''} due now")
        elif upcoming_today > 0:
            nudges.append(f"{upcoming_today} reminder{'s' if upcoming_today != 1 else ''} due today")
    except Exception:
        pass

    # Nudge: recurring error patterns (3+ of same type this month)
    try:
        from omega.bridge import _get_store as _gs

        _store = _gs()
        month_cutoff = (datetime.now(timezone.utc) - __import__("datetime").timedelta(days=30)).isoformat()
        error_rows = _store._conn.execute(
            "SELECT content FROM memories "
            "WHERE event_type = 'error_pattern' AND created_at >= ? "
            "ORDER BY created_at DESC LIMIT 50",
            (month_cutoff,),
        ).fetchall()
        if len(error_rows) >= 3:
            # Group by normalized prefix (first 80 chars, lowered, whitespace-collapsed)
            buckets: dict[str, int] = {}
            for (content,) in error_rows:
                key = re.sub(r"\s+", " ", content[:80].lower()).strip()
                buckets[key] = buckets.get(key, 0) + 1
            top_bucket = max(buckets.items(), key=lambda x: x[1]) if buckets else None
            if top_bucket and top_bucket[1] >= 3:
                nudges.append(
                    f"Pattern: same error {top_bucket[1]}x this month — {top_bucket[0][:60]}"
                )
    except Exception:
        pass

    # Nudge: time-of-day project awareness
    try:
        from omega.bridge import _get_store as _gs_tod

        _store_tod = _gs_tod()
        # Get sessions for the last 14 days, group by hour-of-day + project
        tod_cutoff = (datetime.now(timezone.utc) - __import__("datetime").timedelta(days=14)).isoformat()
        tod_rows = _store_tod._conn.execute(
            "SELECT created_at, metadata FROM memories "
            "WHERE event_type = 'session_summary' AND created_at >= ? "
            "ORDER BY created_at DESC LIMIT 50",
            (tod_cutoff,),
        ).fetchall()
        if len(tod_rows) >= 5:
            current_hour = datetime.now().hour
            # Determine time-of-day bucket
            if 5 <= current_hour < 12:
                tod_label = "morning"
            elif 12 <= current_hour < 17:
                tod_label = "afternoon"
            elif 17 <= current_hour < 22:
                tod_label = "evening"
            else:
                tod_label = "night"

            # Find which projects are most common at this time of day
            project_counts: dict[str, int] = {}
            for (created_at_str, meta_json) in tod_rows:
                try:
                    ca = datetime.fromisoformat(created_at_str.replace("Z", "+00:00"))
                    local_hour = ca.astimezone().hour
                    same_bucket = False
                    if tod_label == "morning" and 5 <= local_hour < 12:
                        same_bucket = True
                    elif tod_label == "afternoon" and 12 <= local_hour < 17:
                        same_bucket = True
                    elif tod_label == "evening" and 17 <= local_hour < 22:
                        same_bucket = True
                    elif tod_label == "night" and (local_hour >= 22 or local_hour < 5):
                        same_bucket = True
                    if same_bucket:
                        meta = json.loads(meta_json) if isinstance(meta_json, str) else (meta_json or {})
                        proj = meta.get("project", "")
                        if proj:
                            proj_name = os.path.basename(proj)
                            project_counts[proj_name] = project_counts.get(proj_name, 0) + 1
                except Exception:
                    continue

            if project_counts:
                top_proj = max(project_counts.items(), key=lambda x: x[1])
                if top_proj[1] >= 3 and project_name != top_proj[0]:
                    nudges.append(f"You typically work on {top_proj[0]} {tod_label}s")
    except Exception:
        pass

    if nudges:
        lines.append("")
        for nudge in nudges[:3]:  # Cap at 3 nudges
            lines.append(f"[NUDGE] {nudge}")

    # --- Section 7b: Auto-surfaced weekly digest (max once per 7 days, 20+ memories) ---
    if memory_count >= 20 and _should_run_periodic("last-digest", 7 * 86400):
        try:
            from omega.bridge import get_weekly_digest

            digest = get_weekly_digest(days=7)
            period_new = digest.get("period_new", 0)
            session_count = digest.get("session_count", 0)
            total = digest.get("total_memories", 0)
            growth_pct = digest.get("growth_pct", 0)
            type_breakdown = digest.get("type_breakdown", {})

            if period_new > 0:
                lines.append("")
                lines.append(f"[WEEKLY] This week: {period_new} memories across {session_count} sessions")
                if type_breakdown:
                    bd_parts = [f"{v} {k.replace('_', ' ')}" for k, v in sorted(type_breakdown.items(), key=lambda x: x[1], reverse=True)[:3]]
                    lines.append(f"  Breakdown: {', '.join(bd_parts)}")
                sign = "+" if growth_pct >= 0 else ""
                lines.append(f"  Trend: {sign}{growth_pct:.0f}% vs prior week | {total} total memories")

                _update_marker("last-digest")
        except Exception as e:
            _log_hook_error("weekly_digest_surface", e)

    # --- Section 8: Footer (maintenance + doctor ok + cloud) ---
    footer_parts = []
    # Maintenance status from markers
    for label, marker_name, cadence in [
        ("backup", "last-backup", 7),
        ("doctor", "last-doctor", 7),
    ]:
        try:
            marker = Path.home() / ".omega" / marker_name
            if marker.exists():
                footer_parts.append(f"{label} ok")
        except Exception:
            pass
    # Cloud sync status
    try:
        secrets_path = Path.home() / ".omega" / "secrets.json"
        if secrets_path.exists():
            pull_marker = Path.home() / ".omega" / "last-cloud-pull"
            if pull_marker.exists():
                last_ts = pull_marker.read_text().strip()
                last = datetime.fromisoformat(last_ts)
                if last.tzinfo is None:
                    last = last.replace(tzinfo=timezone.utc)
                age_days = (datetime.now(timezone.utc) - last).days
                if age_days < 1:
                    footer_parts.append("cloud ok")
                else:
                    footer_parts.append(f"cloud pull: {age_days}d ago")
            else:
                footer_parts.append("cloud pull: never")
    except Exception:
        pass
    if footer_parts:
        lines.append(f"\nMaintenance: {', '.join(footer_parts)}")

    return {"output": "\n".join(lines), "error": None}


def _auto_feedback_on_surfaced(session_id: str):
    """Auto-record feedback for memories surfaced multiple times (likely relevant)."""
    if not session_id:
        return
    json_path = Path.home() / ".omega" / f"session-{session_id}.surfaced.json"
    if not json_path.exists():
        return
    try:
        data = json.loads(json_path.read_text())
        # Count how many times each memory was surfaced across different files.
        # Only record "helpful" for memories surfaced 2+ times (re-surfaced
        # across edits suggests genuine relevance, avoids inflating scores).
        id_counts: dict[str, int] = {}
        for ids in data.values():
            for mid in ids:
                id_counts[mid] = id_counts.get(mid, 0) + 1

        relevant_ids = [mid for mid, count in id_counts.items() if count >= 2]
        if not relevant_ids:
            return

        from omega.bridge import record_feedback

        for mid in relevant_ids[:10]:
            try:
                record_feedback(mid, "helpful", "Auto: surfaced across multiple edits")
            except Exception:
                pass

        json_path.unlink(missing_ok=True)
    except Exception as e:
        _log_hook_error("auto_feedback_surfaced", e)
    finally:
        try:
            if json_path.exists():
                json_path.unlink()
        except Exception:
            pass


def handle_session_stop(payload: dict) -> dict:
    """Generate and store session summary + activity report."""
    session_id = payload.get("session_id", "")
    project = payload.get("project", "")
    entity_id = _resolve_entity(project) if project else None
    lines = []

    # Read surfaced data before auto-feedback cleanup deletes the file
    surfaced_count = 0
    surfaced_unique_ids = 0
    surfaced_unique_files = 0
    try:
        surfaced_json = Path.home() / ".omega" / f"session-{session_id}.surfaced.json"
        if surfaced_json.exists():
            data = json.loads(surfaced_json.read_text())
            surfaced_count = sum(len(ids) for ids in data.values())
            all_ids = set()
            for ids in data.values():
                all_ids.update(ids)
            surfaced_unique_ids = len(all_ids)
            surfaced_unique_files = len(data)
    except Exception:
        pass

    # Auto-feedback for surfaced memories before building summary
    _auto_feedback_on_surfaced(session_id)

    # --- Gather session event counts ---
    counts = {}
    captured = 0
    try:
        from omega.bridge import _get_store

        store = _get_store()
        counts = store.get_session_event_counts(session_id) if session_id else {}
        captured = sum(counts.values()) if counts else 0

        # Clean up surfaced marker
        try:
            marker = Path.home() / ".omega" / f"session-{session_id}.surfaced"
            if marker.exists():
                marker.unlink()
        except Exception:
            pass
    except Exception as e:
        _log_hook_error("session_stop_activity", e)

    # --- Build summary from per-type targeted queries (stored silently) ---
    summary = "Session ended"
    top_decisions: list[str] = []
    try:
        from omega.bridge import query_structured

        decisions = query_structured(
            query_text="decisions made",
            limit=5,
            session_id=session_id,
            project=project,
            event_type="decision",
            entity_id=entity_id,
        )
        errors = query_structured(
            query_text="errors encountered",
            limit=3,
            session_id=session_id,
            project=project,
            event_type="error_pattern",
            entity_id=entity_id,
        )
        tasks = query_structured(
            query_text="completed tasks",
            limit=3,
            session_id=session_id,
            project=project,
            event_type="task_completion",
            entity_id=entity_id,
        )

        parts = []
        if decisions:
            items = [m.get("content", "")[:120] for m in decisions[:3]]
            parts.append(f"Decisions ({len(decisions)}): " + "; ".join(items))
            top_decisions = [m.get("content", "")[:80].replace("\n", " ").strip() for m in decisions[:2]]
        if errors:
            items = [m.get("content", "")[:120] for m in errors[:3]]
            parts.append(f"Errors ({len(errors)}): " + "; ".join(items))
        if tasks:
            items = [m.get("content", "")[:120] for m in tasks[:3]]
            parts.append(f"Tasks ({len(tasks)}): " + "; ".join(items))

        if parts:
            summary = " | ".join(parts)[:600]
        elif decisions or errors or tasks:
            total = len(decisions or []) + len(errors or []) + len(tasks or [])
            summary = f"Session ended with {total} captured memories"
    except Exception as e:
        _log_hook_error("session_stop_summary", e)

    # Store the summary (silent)
    try:
        from omega.bridge import auto_capture

        auto_capture(
            content=f"Session summary: {summary}",
            event_type="session_summary",
            metadata={"source": "session_stop_hook", "project": project},
            session_id=session_id,
            project=project,
            entity_id=entity_id,
        )
    except Exception as e:
        _log_hook_error("session_stop", e)
        return {"output": "\n".join(lines), "error": str(e)}

    # --- Format output: header + details + footer ---
    if captured > 0:
        lines.append(f"## Session complete — {captured} captured, {surfaced_count} surfaced")
        # Type breakdown
        _LABELS = {
            "decision": ("decision", "decisions"),
            "lesson_learned": ("lesson", "lessons"),
            "error_pattern": ("error", "errors"),
        }
        type_parts = []
        other_count = 0
        for key, (singular, plural) in _LABELS.items():
            n = counts.get(key, 0)
            if n:
                type_parts.append(f"{n} {plural if n > 1 else singular}")
        for key, n in counts.items():
            if key not in _LABELS and n > 0:
                other_count += n
        if other_count:
            type_parts.append(f"{other_count} other")
        if type_parts:
            lines.append(f"  Stored: {', '.join(type_parts)}")
        if top_decisions:
            lines.append(f"  Key: {'; '.join(top_decisions)}")
    else:
        lines.append(f"## Session complete — {surfaced_count} memories surfaced")

    # Unique recall stats
    if surfaced_unique_ids > 0:
        lines.append(f"  Recalled: {surfaced_unique_ids} unique memories across {surfaced_unique_files} file{'s' if surfaced_unique_files != 1 else ''}")

    # --- Productivity recap: weekly stats ---
    try:
        from omega.bridge import _get_store as _gs_recap, session_stats as _ss_recap, type_stats as _ts_recap

        store = _gs_recap()
        total_memories = store.node_count()

        # Weekly session count
        all_sessions = _ss_recap()
        weekly_sessions = 0
        try:
            week_cutoff = (datetime.now(timezone.utc) - __import__("datetime").timedelta(days=7)).isoformat()
            weekly_rows = store._conn.execute(
                "SELECT COUNT(DISTINCT session_id) FROM memories "
                "WHERE created_at >= ? AND session_id IS NOT NULL",
                (week_cutoff,),
            ).fetchone()
            weekly_sessions = weekly_rows[0] if weekly_rows else 0
        except Exception:
            pass

        # Weekly memory count
        weekly_memories = 0
        try:
            weekly_mem_row = store._conn.execute(
                "SELECT COUNT(*) FROM memories WHERE created_at >= ?",
                (week_cutoff,),
            ).fetchone()
            weekly_memories = weekly_mem_row[0] if weekly_mem_row else 0
        except Exception:
            pass

        # Prior week memory count (for growth comparison)
        prev_week_memories = 0
        try:
            prev_cutoff = (datetime.now(timezone.utc) - __import__("datetime").timedelta(days=14)).isoformat()
            prev_row = store._conn.execute(
                "SELECT COUNT(*) FROM memories WHERE created_at >= ? AND created_at < ?",
                (prev_cutoff, week_cutoff),
            ).fetchone()
            prev_week_memories = prev_row[0] if prev_row else 0
        except Exception:
            pass

        recap_parts = []
        if weekly_sessions > 1:
            recap_parts.append(f"{weekly_sessions} sessions this week")
        if weekly_memories > 0:
            recap_parts.append(f"{weekly_memories} memories this week")
        recap_parts.append(f"{total_memories} total")
        lines.append(f"  Recap: {', '.join(recap_parts)}")

        # Week-over-week growth
        if prev_week_memories > 0 and weekly_memories > 0:
            growth_pct = ((weekly_memories - prev_week_memories) / prev_week_memories) * 100
            sign = "+" if growth_pct >= 0 else ""
            lines.append(f"  Growth: {sign}{growth_pct:.0f}% vs last week")
    except Exception:
        pass

    # --- Files touched in this session ---
    try:
        from omega.coordination import get_manager as _gm_recap

        mgr = _gm_recap()
        claims = mgr.get_session_claims(session_id)
        file_claims = claims.get("file_claims", [])
        if file_claims:
            fnames = [os.path.basename(f) for f in file_claims[:5]]
            if len(file_claims) > 5:
                fnames.append(f"+{len(file_claims) - 5}")
            lines.append(f"  Files: {', '.join(fnames)}")
    except Exception:
        pass

    # Prune debounce dicts for this session to prevent unbounded growth
    _last_coord_query.pop(session_id, None)
    _pending_urgent.pop(session_id, None)
    _last_surface.clear()
    _error_hashes.clear()
    _error_counts.pop(session_id, None)

    # Auto-sync to cloud (fire-and-forget daemon thread)
    _auto_cloud_sync(session_id)

    return {"output": "\n".join(lines), "error": None}


def handle_surface_memories(payload: dict) -> dict:
    """Surface memories on file edits, capture errors from Bash."""
    tool_name = payload.get("tool_name", "")
    tool_input = payload.get("tool_input", "{}")
    tool_output = payload.get("tool_output") or ""
    if not isinstance(tool_output, str):
        tool_output = json.dumps(tool_output) if isinstance(tool_output, (dict, list)) else str(tool_output)
    session_id = payload.get("session_id", "")
    project = payload.get("project", "")
    entity_id = _resolve_entity(project) if project else None

    # Parse tool input once for all branches
    input_data = _parse_tool_input(payload)

    lines = []

    # Surface memories on file edits
    if tool_name in ("Edit", "Write", "NotebookEdit"):
        file_path = _get_file_path_from_input(input_data)
        if file_path and _debounce_check(_last_surface, file_path, SURFACE_DEBOUNCE_S, _MAX_SURFACE_ENTRIES):
            lines.extend(_surface_for_edit(file_path, session_id, project, entity_id=entity_id))
            _ctx_tags = _ext_to_tags(file_path) or None
            lines.extend(_surface_lessons(file_path, session_id, project, entity_id=entity_id, context_tags=_ctx_tags))

            # Transparent "no results" — show once per session on first edit with no context
            if not lines:
                no_ctx_key = f"_no_ctx_{session_id}"
                if no_ctx_key not in _last_surface:
                    _last_surface[no_ctx_key] = time.monotonic()
                    lines.append(f"[MEMORY] No stored context for {os.path.basename(file_path)} yet")

    # Surface memories on file reads (lightweight — no lessons)
    if tool_name == "Read":
        file_path = input_data.get("file_path", "")
        if file_path and _debounce_check(_last_surface, file_path, SURFACE_DEBOUNCE_S, _MAX_SURFACE_ENTRIES):
            lines.extend(_surface_for_edit(file_path, session_id, project, entity_id=entity_id))

    # Auto-capture errors from Bash failures + track git commits + auto-claim branches
    if tool_name == "Bash" and tool_output:
        recall_lines = _capture_error(tool_output, session_id, project, entity_id=entity_id)
        if recall_lines:
            lines.extend(recall_lines)
        _track_git_commit(tool_input, tool_output, session_id, project)
        _auto_claim_branch(tool_input, session_id, project)

    # Surface peer claims in same directory on edits (multi-agent only, debounced)
    if tool_name in ("Edit", "Write", "NotebookEdit") and session_id and project:
        try:
            edit_path = _get_file_path_from_input(input_data)
            if edit_path:
                edit_dir = os.path.dirname(os.path.abspath(edit_path))
                if _debounce_check(_last_peer_dir_check, edit_dir, PEER_DIR_CHECK_DEBOUNCE_S, _MAX_SURFACE_ENTRIES):
                    from omega.coordination import get_manager

                    mgr = get_manager()
                    sessions = mgr.list_sessions(auto_clean=False)
                    peers = [s for s in sessions
                             if s.get("session_id") != session_id
                             and s.get("project") == project
                             and s.get("status") == "active"]
                    if peers:
                        peer_lines = []
                        for p in peers[:4]:
                            claims = mgr.get_session_claims(p["session_id"])
                            peer_files = claims.get("file_claims", [])
                            same_dir = [os.path.basename(f) for f in peer_files
                                        if os.path.dirname(os.path.abspath(f)) == edit_dir]
                            if same_dir:
                                p_name = _agent_nickname(p["session_id"])
                                flist = ", ".join(same_dir[:4])
                                if len(same_dir) > 4:
                                    flist += f" +{len(same_dir) - 4}"
                                peer_lines.append(f"[PEER] {p_name} has {flist} claimed (same dir as your edit)")
                        if peer_lines:
                            lines.extend(peer_lines[:2])
        except Exception:
            pass  # Fail-open — peer check is best-effort

    # Check for due reminders (debounced — max once per 5 minutes)
    global _last_reminder_check
    now_mono = time.monotonic()
    if _last_reminder_check == 0.0 or now_mono - _last_reminder_check >= REMINDER_CHECK_DEBOUNCE_S:
        _last_reminder_check = now_mono
        try:
            from omega.bridge import get_due_reminders

            due = get_due_reminders(mark_fired=True)
            for r in due[:3]:
                overdue_label = " [OVERDUE]" if r.get("is_overdue") else ""
                lines.append(f"\n[REMINDER]{overdue_label} {r['text']}")
                lines.append(f"  ID: {r['id'][:12]} — dismiss with omega_remind_dismiss")
        except Exception:
            pass  # Fail-open

    return {"output": "\n".join(lines), "error": None}


def _track_surfaced_ids(session_id: str, file_path: str, memory_ids: list):
    """Append surfaced memory IDs to the session's .surfaced.json file."""
    if not session_id or not memory_ids:
        return
    try:
        json_path = Path.home() / ".omega" / f"session-{session_id}.surfaced.json"
        existing = {}
        if json_path.exists():
            existing = json.loads(json_path.read_text())
        prev = existing.get(file_path, [])
        merged = list(set(prev + memory_ids))
        existing[file_path] = merged
        data = json.dumps(existing).encode("utf-8")
        fd = os.open(str(json_path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            os.write(fd, data)
        finally:
            os.close(fd)
    except Exception:
        pass


def _ext_to_tags(file_path: str) -> list:
    """Derive context tags from file extension for re-ranking boost."""
    ext = os.path.splitext(file_path)[1].lower()
    _EXT_MAP = {
        ".py": ["python"],
        ".js": ["javascript"],
        ".ts": ["typescript"],
        ".tsx": ["typescript", "react"],
        ".jsx": ["javascript", "react"],
        ".rs": ["rust"],
        ".go": ["go"],
        ".rb": ["ruby"],
        ".java": ["java"],
        ".swift": ["swift"],
        ".sh": ["bash"],
        ".sql": ["sql"],
        ".md": ["markdown"],
        ".yml": ["yaml"],
        ".yaml": ["yaml"],
        ".json": ["json"],
        ".toml": ["toml"],
        ".css": ["css"],
        ".html": ["html"],
        ".c": ["c"],
        ".cpp": ["c++"],
        ".vue": ["vue", "javascript"],
        ".svelte": ["svelte", "javascript"],
        ".tf": ["terraform"],
        ".graphql": ["graphql"],
        ".prisma": ["prisma"],
        ".env": ["config"],
        ".ini": ["config"],
        ".kt": ["kotlin"],
        ".sc": ["scala"],
        ".ex": ["elixir"],
        ".php": ["php"],
        ".r": ["r"],
    }
    return _EXT_MAP.get(ext, [])


def _apply_confidence_boost(results: list) -> list:
    """Boost relevance scores by capture confidence: high=1.2x, low=0.7x."""
    for r in results:
        confidence = (r.get("metadata") or {}).get("capture_confidence", "medium")
        score = r.get("relevance", 0.0)
        if confidence == "high":
            r["relevance"] = min(1.0, score * 1.2)
        elif confidence == "low":
            r["relevance"] = score * 0.7
    return results


def _surface_for_edit(file_path: str, session_id: str, project: str, *, entity_id: "Optional[str]" = None) -> list[str]:
    """Surface memories related to a file being edited."""
    lines = []
    try:
        from omega.bridge import query_structured
        from omega.sqlite_store import SurfacingContext

        filename = os.path.basename(file_path)
        dirname = os.path.basename(os.path.dirname(file_path))
        context_tags = _ext_to_tags(file_path)
        results = query_structured(
            query_text=f"{filename} {dirname} {file_path}",
            limit=3,
            session_id=session_id,
            project=project,
            context_file=file_path,
            context_tags=context_tags or None,
            filter_tags=context_tags or None,
            entity_id=entity_id,
            surfacing_context=SurfacingContext.FILE_EDIT,
        )
        results = _apply_confidence_boost(results)
        results = [r for r in results if r.get("relevance", 0.0) >= 0.20]
        if results:
            lines.append(f"\n[MEMORY] Relevant context for {filename}:")
            for r in results:
                score = r.get("relevance", 0.0)
                etype = r.get("event_type", "memory")
                preview = r.get("content", "")[:120].replace("\n", " ")
                nid = r.get("id", "")[:8]
                created = r.get("created_at", "")
                age = _relative_time_from_iso(created) if created else ""
                age_part = f" ({age})" if age else ""
                lines.append(f"  [{score:.0%}] {etype}{age_part}: {preview} (id:{nid})")

            # First-recall milestone
            if _check_milestone("first-recall"):
                lines.append("[OMEGA] First memory recalled! Past context is informing this edit.")

            # Track surfaced memory IDs for auto-feedback
            memory_ids = [r.get("id") for r in results if r.get("id")]
            _track_surfaced_ids(session_id, file_path, memory_ids)

            # Traverse: show linked memories (1 hop from top result)
            try:
                top_id = results[0].get("id", "")
                shown_ids = {r.get("id") for r in results}
                if top_id:
                    from omega.bridge import _get_store as _gs

                    _store = _gs()
                    linked = _store.get_related_chain(top_id, max_hops=1, min_weight=0.4)
                    novel = [ln for ln in linked if ln.get("node_id") not in shown_ids][:2]
                    for ln in novel:
                        etype = (ln.get("metadata") or {}).get("event_type", "memory")
                        preview = ln.get("content", "")[:100].replace("\n", " ")
                        lines.append(f"  [linked] {etype}: {preview}")
            except Exception:
                pass

            # Phrase search: exact-match error patterns for this file
            try:
                from omega.bridge import _get_store as _gs2

                _store2 = _gs2()
                exact_errors = _store2.phrase_search(filename, limit=2, event_type="error_pattern")
                shown_ids_updated = {r.get("id") for r in results}
                for err in exact_errors:
                    if err.id not in shown_ids_updated:
                        preview = err.content[:100].replace("\n", " ")
                        lines.append(f"  [exact] error: {preview}")
            except Exception:
                pass
    except Exception as e:
        _log_hook_error("surface_for_edit", e)
    return lines


def _surface_lessons(file_path: str, session_id: str, project: str, *, entity_id: "Optional[str]" = None, context_tags: "Optional[list]" = None) -> list[str]:
    """Surface verified cross-session lessons and peer decisions."""
    lines = []
    try:
        from omega.bridge import get_cross_session_lessons

        filename = os.path.basename(file_path)
        lessons = get_cross_session_lessons(
            task=f"editing {filename}",
            project_path=project,
            exclude_session=session_id,
            limit=2,
            context_file=file_path,
            context_tags=context_tags,
        )
        verified = [lesson for lesson in lessons if lesson.get("verified")]
        if verified:
            lines.append(f"\n[LESSON] Verified wisdom for {filename}:")
            for lesson in verified:
                content = lesson.get("content", "")[:150]
                lines.append(f"  - {content}")
    except Exception as e:
        _log_hook_error("surface_lessons", e)

    # Surface recent peer decisions about this file
    if file_path and session_id:
        try:
            from omega.bridge import query_structured as _qs_peer

            filename = os.path.basename(file_path)
            peer_decisions = _qs_peer(
                query_text=f"decisions about {filename}",
                event_type="decision",
                limit=3,
                context_file=file_path,
                context_tags=context_tags,
                entity_id=entity_id,
            )
            # Filter out own session's decisions and low-relevance results
            for d in peer_decisions:
                meta = d.get("metadata") or {}
                if meta.get("session_id") == session_id:
                    continue
                if d.get("relevance", 0) < 0.5:
                    continue
                content = d.get("content", "")[:100].replace("\n", " ")
                lines.append(f"[PEER-DECISION] {content}")
                break  # Only show 1 to avoid noise
        except Exception:
            pass  # Peer decision surfacing is best-effort

    return lines




def _extract_error_summary(raw_output: str) -> str:
    """Extract a clean error summary from raw tool output.

    For tracebacks: grab the last non-frame line (the actual error).
    For JSON blobs: skip JSON structure, find the error marker line.
    """
    lines = raw_output.strip().split("\n")

    if "Traceback (most recent call last)" in raw_output:
        for line in reversed(lines):
            stripped = line.strip()
            if stripped and not stripped.startswith("File ") and not stripped.startswith("^"):
                return stripped[:300]

    non_json_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith(("{", "[", "}", "]", '"')):
            non_json_lines.append(stripped)
    if non_json_lines:
        error_markers_local = [
            "Error:",
            "ERROR:",
            "error:",
            "FAILED",
            "Failed",
            "SyntaxError:",
            "TypeError:",
            "NameError:",
            "ImportError:",
            "ModuleNotFoundError:",
            "AttributeError:",
            "ValueError:",
            "KeyError:",
            "IndexError:",
            "FileNotFoundError:",
            "fatal:",
            "FATAL:",
            "panic:",
            "command not found",
            "No such file or directory",
            "Permission denied",
            "Connection refused",
        ]
        for line in non_json_lines:
            if any(m in line for m in error_markers_local):
                return line[:300]
        return non_json_lines[0][:300]

    return raw_output[:300]


def _capture_error(tool_output: str, session_id: str, project: str, *, entity_id: "Optional[str]" = None) -> list[str]:
    """Auto-capture error patterns from Bash failures + recall past fixes.

    Session-level dedup: skip if same error pattern already captured.
    Cap at _MAX_ERRORS_PER_SESSION to prevent test-run floods.
    Returns lines for "you've seen this before" recall output.
    """
    if not tool_output:
        return []
    if not isinstance(tool_output, str):
        tool_output = str(tool_output)

    # Cap errors per session
    if _error_counts.get(session_id, 0) >= _MAX_ERRORS_PER_SESSION:
        return []

    error_markers = [
        "Error:",
        "ERROR:",
        "error:",
        "FAILED",
        "Failed",
        "Traceback (most recent call last)",
        "SyntaxError:",
        "TypeError:",
        "NameError:",
        "ImportError:",
        "ModuleNotFoundError:",
        "AttributeError:",
        "ValueError:",
        "KeyError:",
        "IndexError:",
        "FileNotFoundError:",
        "fatal:",
        "FATAL:",
        "panic:",
        "command not found",
        "No such file or directory",
        "Permission denied",
        "Connection refused",
    ]

    if not any(marker in tool_output for marker in error_markers):
        return []

    error_summary = _extract_error_summary(tool_output)

    # Session-level dedup: hash the first 100 chars (normalized)
    error_hash = re.sub(r"\s+", " ", error_summary[:100].lower()).strip()
    if error_hash in _error_hashes:
        return []
    if len(_error_hashes) >= _MAX_ERROR_HASHES:
        return []  # Cap reached — stop tracking new hashes to bound memory
    _error_hashes.add(error_hash)
    _error_counts[session_id] = _error_counts.get(session_id, 0) + 1

    recall_lines: list[str] = []

    # --- Feature: "You've seen this before" — proactive error recall ---
    try:
        from omega.bridge import query_structured
        from omega.sqlite_store import SurfacingContext

        # Search for matching error_pattern and lesson_learned memories
        past_errors = query_structured(
            query_text=error_summary[:200],
            limit=2,
            project=project,
            event_type="error_pattern",
            entity_id=entity_id,
            surfacing_context=SurfacingContext.ERROR_DEBUG,
        )
        past_lessons = query_structured(
            query_text=error_summary[:200],
            limit=2,
            event_type="lesson_learned",
            entity_id=entity_id,
        )

        # Filter to only high-relevance matches (>= 0.35) from previous sessions
        past_matches = []
        for m in (past_errors or []) + (past_lessons or []):
            if m.get("relevance", 0) >= 0.35 and m.get("session_id") != session_id:
                past_matches.append(m)

        if past_matches:
            recall_lines.append("")
            recall_lines.append("[RECALL] You've seen this before:")
            for m in past_matches[:2]:
                etype = m.get("event_type", "memory")
                content = m.get("content", "")[:150].replace("\n", " ").strip()
                rel_time = m.get("relative_time") or ""
                time_note = f" ({rel_time})" if rel_time else ""
                recall_lines.append(f"  [{etype}]{time_note} {content}")
    except Exception as e:
        _log_hook_error("error_recall", e)

    # Store the new error pattern
    try:
        from omega.bridge import auto_capture

        auto_capture(
            content=f"Error: {error_summary}",
            event_type="error_pattern",
            metadata={"source": "auto_capture_hook", "project": project},
            session_id=session_id,
            project=project,
            entity_id=entity_id,
        )
    except Exception as e:
        _log_hook_error("capture_error", e)

    return recall_lines


def _track_git_commit(tool_input: str, tool_output: str, session_id: str, project: str):
    """Detect git commit in Bash output and log to coordination."""
    if not tool_output:
        return
    if not isinstance(tool_output, str):
        tool_output = str(tool_output)

    try:
        input_data = json.loads(tool_input) if isinstance(tool_input, str) else tool_input
    except (json.JSONDecodeError, TypeError):
        return

    command = input_data.get("command", "")
    if "git commit" not in command:
        return

    match = re.search(r"\[[\w/.-]+\s+([0-9a-f]{7,12})\]", tool_output)
    if not match:
        return

    commit_hash = match.group(1)
    msg_match = re.search(r"\[[\w/.-]+\s+[0-9a-f]{7,12}\]\s+(.+)", tool_output)
    message = msg_match.group(1).strip() if msg_match else ""

    branch = _get_current_branch(project)

    try:
        from omega.coordination import get_manager

        mgr = get_manager()
        mgr.log_git_event(
            project=project,
            event_type="commit",
            commit_hash=commit_hash,
            branch=branch,
            message=message,
            session_id=session_id,
        )

        # Auto-release file claims for committed files
        if session_id and project and commit_hash:
            try:
                diff_result = subprocess.run(
                    ["git", "diff-tree", "--no-commit-id", "--name-only", "-r", commit_hash],
                    capture_output=True,
                    text=True,
                    timeout=5,
                    cwd=project,
                )
                if diff_result.returncode == 0 and diff_result.stdout.strip():
                    committed_files = [
                        os.path.join(project, f.strip()) for f in diff_result.stdout.strip().split("\n") if f.strip()
                    ]
                    if committed_files:
                        mgr.release_committed_files(session_id, project, committed_files)
            except Exception:
                pass  # Auto-release is best-effort

    except Exception as e:
        _log_hook_error("track_git_commit", e)


def _auto_claim_branch(tool_input: str, session_id: str, project: str):
    """Auto-claim branch on git checkout/switch (PostToolUse, best-effort)."""
    if not session_id or not project:
        return
    try:
        input_data = json.loads(tool_input) if isinstance(tool_input, str) else tool_input
    except (json.JSONDecodeError, TypeError):
        return

    command = input_data.get("command", "")
    if not any(cmd in command for cmd in ("git checkout", "git switch", "git push")):
        return

    try:
        branch = _get_current_branch(project)
        if not branch or branch == "HEAD":
            return  # Detached HEAD or not a git repo

        from omega.coordination import get_manager

        mgr = get_manager()
        mgr.claim_branch(session_id, project, branch, task="auto-claimed on checkout")
    except Exception:
        pass  # Auto-claim is best-effort


def _clean_task_text(prompt: str) -> str:
    """Delegate to shared implementation in omega.task_utils."""
    from omega.task_utils import clean_task_text
    return clean_task_text(prompt)


def handle_auto_capture(payload: dict) -> dict:
    """Auto-capture decisions and lessons from user prompts (UserPromptSubmit)."""
    # Prefer top-level keys (set by fast_hook.py from parsed stdin JSON).
    # Fall back to re-parsing payload["stdin"] for legacy/direct callers.
    prompt = payload.get("prompt", "")
    if not prompt:
        stdin_data = payload.get("stdin", "")
        if not stdin_data:
            return {"output": "", "error": None}
        try:
            data = json.loads(stdin_data)
        except (json.JSONDecodeError, TypeError):
            return {"output": "", "error": None}
        prompt = data.get("prompt", "")

    session_id = payload.get("session_id", "")
    cwd = payload.get("cwd") or payload.get("project", "")
    entity_id = _resolve_entity(cwd) if cwd else None

    if not prompt:
        return {"output": "", "error": None}

    # Auto-set session task from first prompt (DB as source of truth)
    # Runs before the 20-char guard — short prompts are valid tasks
    if session_id:
        try:
            from omega.coordination import get_manager as _get_mgr_task

            _mgr = _get_mgr_task()
            row = _mgr._conn.execute(
                "SELECT task FROM coord_sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            if row is not None and not row[0]:
                task_text = _clean_task_text(prompt)
                if task_text:
                    with _mgr._lock:
                        _mgr._conn.execute(
                            "UPDATE coord_sessions SET task = ? WHERE session_id = ?",
                            (task_text, session_id),
                        )
                        _mgr._conn.commit()
        except Exception:
            pass  # Auto-task is best-effort

    # Short prompts: task is set above, but skip decision/lesson/preference capture
    if len(prompt) < 20:
        return {"output": "", "error": None}

    # --- Surface peer work on planning prompts (coordination awareness) ---
    coord_output = ""
    _planning_patterns = [
        r"\bwhat.?s?\s+next\b",
        r"\bwhat\s+should\s+(?:i|we)\b",
        r"\bwhat\s+(?:to|can)\s+(?:do|work)\b",
        r"\bnext\s+(?:step|task|priority)\b",
        r"\bpriorities?\b",
        r"\broadmap\b",
        r"\bwhat\s+(?:remains?|is\s+left)\b",
    ]
    prompt_lower_early = prompt.lower()
    is_planning = any(re.search(pat, prompt_lower_early) for pat in _planning_patterns)
    if is_planning and session_id and cwd:
        now = time.monotonic()
        if session_id not in _last_coord_query or now - _last_coord_query[session_id] >= COORD_QUERY_DEBOUNCE_S:
            try:
                from omega.coordination import get_manager as _get_mgr_coord

                mgr = _get_mgr_coord()
                sessions = mgr.list_sessions(auto_clean=False)
                peers = [
                    s for s in sessions
                    if s.get("session_id") != session_id and s.get("project") == cwd
                ][:4]
                if peers:
                    # Fetch in-progress coord tasks for this project
                    in_progress_tasks = []
                    try:
                        in_progress_tasks = mgr.list_tasks(project=cwd, status="in_progress")
                    except Exception:
                        pass
                    task_by_session: dict[str, dict] = {}
                    for t in in_progress_tasks:
                        sid = t.get("session_id")
                        if sid and sid not in task_by_session:
                            task_by_session[sid] = t

                    coord_lines = [f"[COORD] {len(peers)} peer{'s' if len(peers) != 1 else ''} active on this project:"]
                    for p in peers:
                        p_sid = p["session_id"]
                        p_name = _agent_nickname(p_sid)
                        # Prefer coord_task over session.task
                        ct = task_by_session.get(p_sid)
                        if ct:
                            pct = f" [{ct['progress']}%]" if ct.get("progress") else ""
                            p_task = f"#{ct['id']} {ct['title'][:40]}{pct}"
                        else:
                            p_task = (p.get("task") or "idle")[:50]
                        # File claims
                        p_files = ""
                        try:
                            claims = mgr.get_session_claims(p_sid)
                            file_claims = claims.get("file_claims", [])
                            if file_claims:
                                fnames = [os.path.basename(f) for f in file_claims[:3]]
                                if len(file_claims) > 3:
                                    fnames.append(f"+{len(file_claims) - 3}")
                                p_files = f" [{', '.join(fnames)}]"
                        except Exception:
                            pass
                        coord_lines.append(f"  {p_name}: {p_task}{p_files}")
                    coord_output = "\n".join(coord_lines)
                _last_coord_query[session_id] = now
            except Exception:
                pass  # Coordination query is best-effort

    # Auto-classify intent (router integration)
    router_output = ""
    classified_intent = None
    try:
        from omega.router.classifier import classify_intent

        intent, confidence = classify_intent(prompt)
        if confidence >= 0.6:
            classified_intent = intent
            router_output = f"[ROUTER] Intent: {intent} ({confidence:.0%})"
    except ImportError:
        pass  # Router is optional
    except Exception:
        pass  # Classification is best-effort

    # Preference pattern matching (checked first — highest priority)
    preference_patterns = [
        r"\bi\s+(?:prefer|like|love|enjoy|favor|favour)\s+\w",
        r"\bmy\s+(?:preference|favorite|favourite|default)\b",
        r"\balways\s+use\b",
        r"\bi\s+(?:want|need)\s+(?:it|things?|everything)\s+(?:in|with|to\s+be)\b",
        r"\bremember\s+(?:that\s+)?i\s+(?:prefer|like|want|use|need)\b",
        r"\bdon'?t\s+(?:ever\s+)?(?:use|suggest|recommend)\b",
        r"\bi\s+(?:hate|dislike|avoid)\b",
    ]

    # Decision pattern matching
    decision_patterns = [
        r"\blet'?s?\s+(?:go\s+with|use|switch\s+to|stick\s+with|move\s+to)\b",
        r"\bi\s+(?:decided?|chose|picked|went\s+with)\b",
        r"\bwe\s+(?:should|will|are\s+going\s+to)\s+(?:use|go\s+with|switch|adopt|implement)\b",
        r"\b(?:decision|approach|strategy):\s*\S",
        r"\binstead\s+of\s+\S+[,\s]+(?:use|let'?s|we'?ll)\b",
        r"\bfrom\s+now\s+on\b",
        r"\bremember\s+(?:that|this)\b",
    ]

    # Lesson pattern matching
    lesson_patterns = [
        r"\bi\s+learned\s+that\b",
        r"\bturns?\s+out\b",
        r"\bthe\s+trick\s+is\b",
        r"\bnote\s+to\s+self\b",
        r"\btil\b|\btoday\s+i\s+learned\b",
        r"\bthe\s+fix\s+was\b",
        r"\bthe\s+problem\s+was\b",
        r"\bdon'?t\s+forget\b",
        r"\bimportant:\s*\S",
        r"\bkey\s+(?:insight|takeaway|learning)\b",
        r"\bnever\s+(?:again|do|use)\b",
        r"\balways\s+(?:make\s+sure|remember|check)\b",
    ]

    prompt_lower = prompt.lower()
    is_preference = any(re.search(pat, prompt_lower) for pat in preference_patterns)
    is_decision = any(re.search(pat, prompt_lower) for pat in decision_patterns)
    is_lesson = any(re.search(pat, prompt_lower) for pat in lesson_patterns)

    if not is_preference and not is_decision and not is_lesson:
        # Pass through any accumulated coord/router output even if no capture
        passthrough = "\n".join(filter(None, [coord_output, router_output]))
        return {"output": passthrough, "error": None}

    # Preference > Decision > Lesson priority
    if is_preference:
        event_type = "user_preference"
        content_prefix = "Preference"
    elif is_decision:
        event_type = "decision"
        content_prefix = "Decision"
    else:
        event_type = "lesson_learned"
        content_prefix = "Lesson"

    # Lesson quality gate: min 50 chars, >= 7 words
    # Pattern match already signals intent — no secondary tech signal required
    if event_type == "lesson_learned":
        if len(prompt) < 50 or len(prompt.split()) < 7:
            return {"output": "", "error": None}

    try:
        from omega.bridge import auto_capture

        meta = {"source": "auto_capture_hook", "project": cwd}
        if classified_intent:
            meta["intent"] = classified_intent
        auto_capture(
            content=f"{content_prefix}: {prompt[:500]}",
            event_type=event_type,
            metadata=meta,
            session_id=session_id,
            project=cwd,
            entity_id=entity_id,
        )
    except Exception as e:
        _log_hook_error("auto_capture", e)
        return {"output": "\n".join(filter(None, [coord_output, router_output])), "error": None}

    # User-visible confirmation of what was captured
    capture_line = f"[CAPTURED] {content_prefix.lower()}: {prompt[:80].replace(chr(10), ' ').strip()}"
    combined = "\n".join(filter(None, [capture_line, coord_output, router_output]))
    return {"output": combined, "error": None}


def _get_current_branch(project: str) -> str | None:
    """Get the current git branch name."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=project,
        )
        return result.stdout.strip() if result.returncode == 0 else None
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return None


# ---------------------------------------------------------------------------
# Handler dispatch table — core handlers are always available;
# additional handlers can be provided by plugins.
# ---------------------------------------------------------------------------

# Core memory handlers — always shipped with omega-memory
_CORE_HOOK_HANDLERS = {
    "session_start": handle_session_start,
    "session_stop": handle_session_stop,
    "surface_memories": handle_surface_memories,
    "auto_capture": handle_auto_capture,
}

# Build the dispatch table: core + plugins
HOOK_HANDLERS = dict(_CORE_HOOK_HANDLERS)

# Discover plugin-provided hook handlers
try:
    from omega.plugins import discover_plugins

    for _plugin in discover_plugins():
        if _plugin.HOOK_HANDLERS:
            HOOK_HANDLERS.update(_plugin.HOOK_HANDLERS)
except Exception:
    pass


def register_hook_handler(name: str, handler):
    """Register a hook handler at runtime (for plugins)."""
    HOOK_HANDLERS[name] = handler


# ---------------------------------------------------------------------------
# UDS Server
# ---------------------------------------------------------------------------


async def handle_connection(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
    """Handle a single hook client connection."""
    t0 = time.monotonic()
    hook_name = "unknown"
    try:
        # Read until EOF — client calls shutdown(SHUT_WR) after sendall()
        chunks = []
        while True:
            chunk = await asyncio.wait_for(reader.read(65536), timeout=10.0)
            if not chunk:
                break
            chunks.append(chunk)
        data = b"".join(chunks)
        if not data:
            writer.close()
            return

        request = json.loads(data.decode("utf-8").strip())

        # Batch mode: {"hooks": ["a", "b", ...], ...}
        # Single mode: {"hook": "a", ...}
        hook_names = request.pop("hooks", None)
        if hook_names:
            hook_name = "+".join(hook_names)
            loop = asyncio.get_running_loop()
            results = []
            for name in hook_names:
                handler = HOOK_HANDLERS.get(name)
                if not handler:
                    results.append({"output": "", "error": f"Unknown hook: {name}"})
                else:
                    r = await loop.run_in_executor(None, handler, request)
                    results.append(r)
                    # Short-circuit on block — skip remaining hooks
                    if r.get("exit_code"):
                        break
            response = {"results": results}
        else:
            hook_name = request.pop("hook", "unknown")
            handler = HOOK_HANDLERS.get(hook_name)
            if not handler:
                response = {"output": "", "error": f"Unknown hook: {hook_name}"}
            else:
                # Run handler in executor to avoid blocking the event loop
                loop = asyncio.get_running_loop()
                response = await loop.run_in_executor(None, handler, request)

        writer.write(json.dumps(response).encode("utf-8"))
        await writer.drain()
    except asyncio.TimeoutError:
        try:
            writer.write(json.dumps({"output": "", "error": "timeout"}).encode("utf-8"))
            await writer.drain()
        except Exception:
            pass
    except Exception as e:
        _log_hook_error(f"connection/{hook_name}", e)
        try:
            writer.write(json.dumps({"output": "", "error": str(e)}).encode("utf-8"))
            await writer.drain()
        except Exception:
            pass
    finally:
        elapsed_ms = (time.monotonic() - t0) * 1000
        _log_timing(hook_name, elapsed_ms)
        try:
            writer.close()
            await writer.wait_closed()
        except Exception:
            pass


_hook_server: asyncio.Server | None = None


async def start_hook_server() -> asyncio.Server | None:
    """Start the UDS hook server. Returns the server instance."""
    global _hook_server

    # Ensure directory exists
    SOCK_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Remove stale socket from previous run
    if SOCK_PATH.exists():
        SOCK_PATH.unlink()

    try:
        _hook_server = await asyncio.start_unix_server(handle_connection, path=str(SOCK_PATH))
        # Make socket accessible
        SOCK_PATH.chmod(0o600)
        logger.info("Hook server listening on %s", SOCK_PATH)
        return _hook_server
    except Exception as e:
        logger.error("Failed to start hook server: %s", e)
        return None


async def stop_hook_server(srv: asyncio.Server | None = None):
    """Stop the hook server and clean up socket.

    Only deletes the socket file if this process owns the server,
    to avoid breaking another MCP server's active socket.
    """
    global _hook_server
    server = srv or _hook_server
    if server:
        server.close()
        await server.wait_closed()
        _hook_server = None

        # Only unlink if we were the ones serving
        if SOCK_PATH.exists():
            try:
                SOCK_PATH.unlink()
            except Exception:
                pass
