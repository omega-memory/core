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
_last_heartbeat: dict[str, float] = {}  # session_id -> monotonic timestamp
_last_claim: dict[tuple[str, str], float] = {}  # (session_id, file_path) -> timestamp
_last_surface: dict[str, float] = {}  # file_path -> timestamp (capped at 500)
_MAX_SURFACE_ENTRIES = 500
_last_overlap_notify: dict[tuple[str, str, str], float] = {}  # (sid, other_sid, file) -> timestamp
OVERLAP_NOTIFY_DEBOUNCE_S = 300.0  # 5 minutes
_last_block_notify: dict[tuple[str, str, str], float] = {}  # (blocked_sid, owner_sid, file) -> timestamp
BLOCK_NOTIFY_DEBOUNCE_S = 300.0  # 5 minutes
_last_peer_dir_check: dict[str, float] = {}  # dir_path -> timestamp
PEER_DIR_CHECK_DEBOUNCE_S = 300.0  # 5 minutes
_last_coord_query: dict[str, float] = {}  # session_id -> monotonic timestamp
COORD_QUERY_DEBOUNCE_S = 120.0  # 2 minutes
_last_reminder_check: float = 0.0
REMINDER_CHECK_DEBOUNCE_S = 300.0  # 5 minutes

_last_deadlock_push: dict[str, float] = {}  # cycle_hash -> monotonic timestamp
DEADLOCK_PUSH_DEBOUNCE_S = 600.0  # 10 minutes — don't spam same cycle

HEARTBEAT_DEBOUNCE_S = 30.0
CLAIM_DEBOUNCE_S = 30.0
SURFACE_DEBOUNCE_S = 5.0

_heartbeat_count: dict[str, int] = {}  # session_id -> call count (for inbox surfacing)
_peer_snapshot: dict[str, set] = {}  # session_id -> set of known peer session_ids
_session_intent: dict[str, str] = {}  # session_id -> latest classified intent

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
    try:
        from omega.router.engine import OmegaRouter as _WelcomeRouter

        _wr = _WelcomeRouter()
        _priority = getattr(_wr, "priority_mode", "balanced")
        health_line += f" | **Router:** {_priority}"
    except Exception:
        pass
    lines.append(health_line)

    # Graph health ratio
    _graph_ratio = 0.0
    try:
        from omega.bridge import _get_store as _gs_graph

        _graph_store = _gs_graph()
        _node_count = _graph_store.count()
        _edge_count = _graph_store.edge_count()
        if _node_count > 0:
            _graph_ratio = _edge_count / _node_count
            _graph_label = "rich" if _graph_ratio >= 1.5 else ("good" if _graph_ratio >= 0.5 else "sparse")
            lines.append(f"**Graph:** {_graph_label} ({_edge_count:,} edges, {_graph_ratio:.1f}x)")
    except Exception:
        pass

    # Profile summary (field counts across categories)
    try:
        from omega.profile.engine import get_profile_engine

        _pe = get_profile_engine()
        with _pe._lock:
            _profile_rows = _pe._conn.execute(
                "SELECT COUNT(*) as cnt FROM secure_profile"
            ).fetchone()
        _profile_count = _profile_rows["cnt"] if _profile_rows else 0
        if _profile_count > 0:
            lines.append(f"**Profile:** {_profile_count} encrypted field(s) stored")
    except Exception:
        pass

    # Proactive maintenance suggestion when graph is sparse
    if _graph_ratio > 0 and _graph_ratio < 0.5:
        lines.append("[MAINTENANCE] Graph connectivity is sparse — consider running omega_compact to consolidate related memories")

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

        # Fallback: if no 2+ surfacings but multiple unique memories were surfaced
        # once during active editing (multiple files edited), give lightweight feedback
        if not relevant_ids and len(data) >= 2:
            once_ids = [mid for mid, count in id_counts.items() if count == 1]
            relevant_ids = once_ids[:5]
            feedback_reason = "Auto: surfaced during active multi-file editing"
        else:
            feedback_reason = "Auto: surfaced across multiple edits"

        if not relevant_ids:
            return

        from omega.bridge import record_feedback

        for mid in relevant_ids[:10]:
            try:
                record_feedback(mid, "helpful", feedback_reason)
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
        from omega.bridge import _get_store as _gs_recap, session_stats as _ss_recap

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
    _last_heartbeat.pop(session_id, None)
    _heartbeat_count.pop(session_id, None)
    stale_claims = [k for k in _last_claim if k[0] == session_id]
    for k in stale_claims:
        del _last_claim[k]
    stale_overlaps = [k for k in _last_overlap_notify if k[0] == session_id]
    for k in stale_overlaps:
        del _last_overlap_notify[k]
    _last_coord_query.pop(session_id, None)
    _pending_urgent.pop(session_id, None)
    _session_intent.pop(session_id, None)
    _last_surface.clear()
    _error_hashes.clear()
    _error_counts.pop(session_id, None)

    # Auto-sync to cloud (fire-and-forget daemon thread)
    _auto_cloud_sync(session_id)

    return {"output": "\n".join(lines), "error": None}


def handle_coord_session_start(payload: dict) -> dict:
    """Register agent session + git sync check + session resume."""
    session_id = payload.get("session_id", "")
    project = payload.get("project", "")
    if not session_id:
        return {"output": "", "error": None}

    lines = []
    peer_count = 0
    try:
        from omega.coordination import get_manager

        mgr = get_manager()
        mgr.list_sessions()  # Force-clean stale sessions (bypasses rate limit)
        result = mgr.register_session(
            session_id=session_id,
            pid=os.getpid(),
            project=project,
        )
        peer_count = result.get("peers_on_project", 0)

        # --- [!] Alerts: file/branch conflicts from peers ---
        if peer_count > 0:
            try:
                status = mgr.get_status()
                conflicts = status.get("conflicts", [])
                for conflict in conflicts[:3]:
                    file_path = conflict.get("file", conflict.get("file_path", "unknown"))
                    owner_sid = conflict.get("owner", conflict.get("session_id", "another agent"))
                    owner_name = _agent_nickname(owner_sid) if owner_sid != "another agent" else owner_sid
                    lines.append(f"[!] File conflict: {file_path} claimed by {owner_name}")
            except Exception:
                pass  # Conflict check is best-effort

            # --- [!] Alerts: uncommitted git files that peers own ---
            try:
                if project:
                    git_result = subprocess.run(
                        ["git", "status", "--porcelain"],
                        capture_output=True,
                        text=True,
                        timeout=5,
                        cwd=project,
                    )
                    if git_result.returncode == 0 and git_result.stdout.strip():
                        uncommitted = set()
                        for line in git_result.stdout.strip().split("\n"):
                            if line.strip():
                                parts = line[3:].split(" -> ")
                                rel_path = parts[-1].strip()
                                uncommitted.add(os.path.abspath(os.path.join(project, rel_path)))

                        peer_owned: dict[str, list[str]] = {}
                        for f in status.get("files", []):
                            f_path = f.get("path", "")
                            f_sid = f.get("session_id", "")
                            if f_sid != session_id and f_path in uncommitted:
                                name = _agent_nickname(f_sid)
                                peer_owned.setdefault(name, []).append(os.path.basename(f_path))

                        for name, fnames in peer_owned.items():
                            flist = ", ".join(fnames[:4])
                            if len(fnames) > 4:
                                flist += f" +{len(fnames) - 4}"
                            lines.append(
                                f"[!] {len(fnames)} uncommitted file{'s' if len(fnames) != 1 else ''}"
                                f" owned by {name}: {flist} — do NOT commit these"
                            )
            except Exception:
                pass  # Uncommitted file check is best-effort

            # Check for unread messages → feed into [TODO] section
            try:
                unread = mgr.get_unread_count(session_id)
            except Exception:
                unread = 0
        else:
            unread = 0

        # --- Git sync check → [!] alerts ---
        git_lines = _check_git_sync(session_id, project, mgr)
        lines.extend(git_lines)

        # Session intent announce + branch claim on start (silent)
        try:
            branch = _get_current_branch(project) if project else None
            if branch and branch not in ("main", "master", "develop", "release"):
                mgr.announce_intent(
                    session_id=session_id,
                    description=f"Working on branch {branch}",
                    intent_type="session_start",
                    target_files=[],
                    target_branch=branch,
                    ttl_minutes=120,
                )
                mgr.claim_branch(session_id, project, branch, task="session start")
        except Exception:
            pass  # Intent/branch claim is best-effort

        # --- [RESUME] from predecessor session ---
        resume_lines = _session_resume(session_id, project, mgr)
        lines.extend(resume_lines)

        # --- [HANDOFF] from predecessor's complete message ---
        try:
            handoff_msgs = mgr.check_inbox(session_id, unread_only=True, msg_type="complete", limit=1)
            if handoff_msgs:
                msg = handoff_msgs[0]
                from_sid = msg.get("from_session") or "unknown"
                from_name = _agent_nickname(from_sid)
                subj = (msg.get("subject") or "")[:80]
                body = (msg.get("body") or "")[:500]
                handoff_line = f"\n[HANDOFF] From {from_name}: {subj}"
                if body:
                    for bl in body.strip().split("\n")[:10]:
                        handoff_line += f"\n  {bl.strip()}"
                lines.append(handoff_line)
        except Exception:
            pass  # Handoff surfacing is best-effort

        # --- [CONTINUE] recently reassigned tasks (have progress > 0) ---
        try:
            all_pending = mgr.list_tasks(project=project, status="pending")
            continued = [t for t in all_pending if t.get("progress", 0) > 0]
            if continued:
                t = continued[0]  # highest priority (list_tasks already sorted)
                pct = f" [{t['progress']}%]" if t.get("progress") else ""
                lines.append(
                    f"\n[CONTINUE] Task #{t['id']} \"{t['title']}\"{pct} was in progress "
                    f"when last session ended — claim with omega_task_next"
                )
        except Exception:
            all_pending = None

        # --- [TODO] pending tasks + unread messages ---
        try:
            tasks = all_pending if all_pending is not None else mgr.list_tasks(project=project, status="pending")
        except Exception:
            tasks = []

        todo_parts = []
        if tasks:
            todo_parts.append(f"{len(tasks)} pending tasks")
        if unread > 0:
            todo_parts.append(f"{unread} unread msg")
        if todo_parts:
            lines.append(f"\n[TODO] {' | '.join(todo_parts)} — query omega_tasks_list / omega_inbox for full list")
            # Show the highest-priority next task with auto-claim hint
            if tasks:
                unblocked = [t for t in tasks if not t.get("blocked")]
                show_task = unblocked[0] if unblocked else max(tasks, key=lambda t: t.get("priority", 0))
                prio = f" P{show_task['priority']}" if show_task.get("priority") else ""
                hint = " — use omega_task_next to auto-claim" if unblocked else ""
                lines.append(f"  NEXT: #{show_task['id']}{prio} {show_task['title']}{hint}")

        # Router warmup (silent — only errors become [!] alerts)
        try:
            from omega.router.classifier import warm_up

            warm_up()
        except ImportError:
            pass  # Router is optional
        except Exception as e_r:
            _log_hook_error("router_warmup", e_r)

        # --- Coordination celebration (multi-agent success) ---
        if peer_count > 0:
            try:
                status = mgr.get_status()
                n_conflicts = len(status.get("conflicts", []))
                n_deadlocks = len(status.get("deadlocks", []))
                if n_conflicts == 0 and n_deadlocks == 0:
                    completed_tasks = mgr.list_tasks(status="completed")
                    today = datetime.now(timezone.utc).date().isoformat()
                    today_done = [t for t in completed_tasks if (t.get("updated_at") or "")[:10] == today]
                    if today_done:
                        lines.append(
                            f"\n[COORD] {peer_count + 1} agents, "
                            f"{len(today_done)} task{'s' if len(today_done) != 1 else ''} "
                            f"completed today with zero conflicts"
                        )
            except Exception:
                pass

        # --- Footer: rich [COORD] team roster (cross-project) ---
        try:
            sessions = mgr.list_sessions(auto_clean=False)
            peers = [s for s in sessions if s.get("session_id") != session_id][:6]
            if peers:
                # Fetch in-progress coord tasks to prefer over session.task
                _roster_tasks: dict[str, dict] = {}
                try:
                    for _rt in mgr.list_tasks(project=project, status="in_progress"):
                        _rt_sid = _rt.get("session_id")
                        if _rt_sid and _rt_sid not in _roster_tasks:
                            _roster_tasks[_rt_sid] = _rt
                except Exception:
                    pass
                lines.append(f"\n[COORD] {len(peers)} peer{'s' if len(peers) != 1 else ''} active:")
                for p in peers:
                    p_name = _agent_nickname(p["session_id"])
                    # Prefer coord_task over session.task
                    _ct = _roster_tasks.get(p["session_id"])
                    if _ct:
                        _pct = f" [{_ct['progress']}%]" if _ct.get("progress") else ""
                        p_task = f"#{_ct['id']} {_ct['title'][:40]}{_pct}"
                    else:
                        p_task = (p.get("task") or "idle")[:50]
                    # Get claimed files and branch
                    p_files = ""
                    p_branch = ""
                    try:
                        claims = mgr.get_session_claims(p["session_id"])
                        file_claims = claims.get("file_claims", [])
                        if file_claims:
                            fnames = [os.path.basename(f) for f in file_claims[:2]]
                            if len(file_claims) > 2:
                                fnames.append(f"+{len(file_claims) - 2}")
                            p_files = f" [{', '.join(fnames)}]"
                        branch_claims = claims.get("branch_claims", [])
                        if branch_claims:
                            p_branch = f" ({branch_claims[0]})"
                    except Exception:
                        pass
                    # Project label when different from current session
                    p_proj_label = ""
                    p_project = p.get("project") or ""
                    if p_project and p_project != project:
                        p_proj_label = f" [{os.path.basename(p_project)}]"
                    # Heartbeat age
                    age = _format_age(p.get("last_heartbeat", ""))
                    age_str = f" — {age}" if age else ""
                    lines.append(f"  {p_name}: {p_task}{p_proj_label}{p_files}{p_branch}{age_str}")
        except Exception:
            if peer_count > 0:
                lines.append(f"\n{peer_count} peer(s) active — query omega_sessions_list for details")

    except Exception as e:
        _log_hook_error("coord_session_start", e)
        return {"output": "", "error": str(e)}

    return {"output": "\n".join(lines), "error": None}


def _check_git_sync(session_id: str, project: str, mgr) -> list[str]:
    """Detect upstream commits from uncoordinated agents."""
    lines = []
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=project,
        )
        if result.returncode != 0:
            return lines

        subprocess.run(
            ["git", "fetch", "origin", "--quiet"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=project,
        )

        branch = _get_current_branch(project) or "main"

        upstream_result = subprocess.run(
            ["git", "log", f"HEAD..origin/{branch}", "--oneline", "--no-decorate", "-20"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=project,
        )
        if upstream_result.returncode != 0 or not upstream_result.stdout.strip():
            return lines

        upstream_lines = upstream_result.stdout.strip().split("\n")
        upstream_hashes = [line.split()[0] for line in upstream_lines if line.strip()]
        if not upstream_hashes:
            return lines

        untracked = mgr.detect_untracked_commits(project, upstream_hashes)

        for line in upstream_lines:
            parts = line.split(None, 1)
            if parts:
                mgr.log_git_event(
                    project=project,
                    event_type="upstream_detected",
                    commit_hash=parts[0],
                    branch=branch,
                    message=parts[1] if len(parts) > 1 else "",
                )

        lines.append(
            f"[!] {len(upstream_hashes)} upstream commit(s) on origin/{branch} — run `git pull` before editing"
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    except Exception as e:
        _log_hook_error("git_sync_check", e)
    return lines


def _session_resume(session_id: str, project: str, mgr) -> list[str]:
    """Check for predecessor session snapshots with enriched context."""
    lines = []
    try:
        if not project:
            return lines
        snapshots = mgr.recover_session(project)
        if not snapshots:
            snapshots = []  # Fall through to checkpoint surfacing

        if snapshots:
            snap = snapshots[0]
            meta = snap.get("metadata") or {}

            # Compute how long ago the predecessor ended
            age = _format_age(snap.get("created_at", ""))
            age_str = f" ended {age}" if age else ""

            # Build compact single-line [RESUME]
            task_desc = (snap.get("task") or "")[:60]
            task_part = f'"{task_desc}"' if task_desc else "previous session"

            # Get top decisions as Key: summary
            key_parts = []
            try:
                from omega.bridge import query_structured

                decisions = query_structured(
                    query_text="decisions made",
                    limit=5,
                    session_id=snap["session_id"],
                    project=project,
                    event_type="decision",
                )
                for d in decisions or []:
                    content = d.get("content", "")
                    # Strip auto-capture prefixes
                    for prefix in ("Plan/decision captured: ", "Decision: "):
                        if content.startswith(prefix):
                            content = content[len(prefix) :]
                    # Skip JSON blobs
                    stripped = content.lstrip()
                    if stripped.startswith(("{", "[", '"filePath')):
                        continue
                    # Take first meaningful line only
                    first_line = content.split("\n")[0].strip()
                    if first_line and len(first_line) > 10:
                        key_parts.append(first_line[:80])
                    if len(key_parts) >= 2:
                        break
            except Exception:
                pass  # Decision surfacing is best-effort

            key_str = f" | Key: {'; '.join(key_parts)}" if key_parts else ""
            lines.append(f"[RESUME] {task_part}{age_str}{key_str}")
    except Exception as e:
        _log_hook_error("session_resume", e)

    # Surface recent checkpoints for this project
    try:
        if project:
            from omega.bridge import query_structured

            checkpoints = query_structured(
                query_text="checkpoint",
                limit=3,
                event_type="checkpoint",
                project=project,
            )
            if checkpoints:
                checkpoints.sort(key=lambda c: c.get("created_at", ""), reverse=True)
                latest = checkpoints[0]
                meta = latest.get("metadata", {})
                cp_data = meta.get("checkpoint_data", {})
                task_title = cp_data.get("task_title", "Unknown task")
                next_steps = cp_data.get("next_steps", "")

                # Compute age
                age = _format_age(latest.get("created_at", ""))
                cp_age = f" saved {age}" if age else ""

                cp_line = f'[CHECKPOINT] "{task_title}"{cp_age}'
                if next_steps:
                    cp_line += f" | Next: {next_steps[:80]}"
                cp_line += " — call `omega_resume_task` for full context"
                lines.append(cp_line)
    except Exception:
        pass  # Fail-open

    return lines


def handle_coord_session_stop(payload: dict) -> dict:
    """Deregister session and release all claims."""
    session_id = payload.get("session_id", "")
    if not session_id:
        return {"output": "", "error": None}

    lines = []
    try:
        from omega.coordination import get_manager

        mgr = get_manager()
        project = payload.get("project", "")
        if project:
            mgr.enrich_session_metadata(session_id, project)

        # Count peers before deregistering (for footer)
        peer_count = 0
        try:
            sessions = mgr.list_sessions(auto_clean=True)
            peer_count = sum(1 for s in sessions if s.get("session_id") != session_id)
        except Exception:
            pass

        # Broadcast structured handoff message for successor sessions (silent)
        try:
            summary = "Session ended"
            try:
                from omega.bridge import _get_store

                store = _get_store()
                counts = store.get_session_event_counts(session_id) if session_id else {}
                if counts:
                    total = sum(counts.values())
                    summary = f"Session ended ({total} memories captured)"
            except Exception:
                pass

            handoff_parts = [f"## Session Summary\n{summary[:300]}"]

            # Decisions made this session
            try:
                from omega.bridge import query_structured as _qs_handoff

                decisions = _qs_handoff(
                    query_text="decisions made",
                    session_id=session_id,
                    event_type="decision",
                    limit=5,
                )
                if decisions:
                    handoff_parts.append("## Decisions")
                    for d in decisions:
                        handoff_parts.append(f"- {d.get('content', '')[:120]}")
            except Exception:
                pass

            # Errors/blockers encountered
            try:
                from omega.bridge import query_structured as _qs_errors

                errors = _qs_errors(
                    query_text="errors encountered",
                    session_id=session_id,
                    event_type="error_pattern",
                    limit=3,
                )
                if errors:
                    handoff_parts.append("## Blockers")
                    for e in errors:
                        handoff_parts.append(f"- {e.get('content', '')[:120]}")
            except Exception:
                pass

            # Incomplete tasks owned by this session
            incomplete = []
            try:
                all_tasks = mgr.list_tasks(project=project, status="in_progress")
                incomplete = [t for t in all_tasks if t.get("session_id") == session_id]
                if incomplete:
                    handoff_parts.append("## Incomplete Work")
                    for t in incomplete:
                        handoff_parts.append(f"- Task #{t['id']}: {t['title']}")
            except Exception:
                pass

            handoff_body = "\n".join(handoff_parts)[:2000]

            mgr.send_message(
                from_session=session_id,
                subject=f"Handoff: {summary[:60]}",
                msg_type="complete",
                project=project or None,
                body=handoff_body,
                ttl_minutes=120,
            )
        except Exception:
            pass  # Handoff message is best-effort

        # Reassign orphaned tasks to pending so next session can claim them
        reassigned = []
        try:
            reassigned = mgr.reassign_orphaned_tasks(session_id)
            if reassigned:
                task_list = ", ".join(f"#{t['id']} {t['title']}" for t in reassigned[:3])
                lines.append(f"  Tasks returned to queue: {task_list}")
        except Exception:
            pass  # Reassign is best-effort

        # Task state for output (completed + continuing)
        my_completed = []
        try:
            completed_tasks = mgr.list_tasks(project=project, status="completed")
            my_completed = [t for t in completed_tasks if t.get("session_id") == session_id]
            for t in my_completed[:2]:
                lines.append(f"  Done: #{t['id']} {t['title']}")
            for t in reassigned[:2]:
                progress = t.get("progress", 0)
                prog_str = f" ({progress}%)" if progress else ""
                lines.append(f"  Continuing: #{t['id']} {t['title']}{prog_str} — returned to queue for next session")
        except Exception:
            pass

        # [COORD] summary block — files, messages, tasks
        try:
            coord_parts = []
            # Files claimed this session
            claims = mgr.get_session_claims(session_id)
            file_claims = claims.get("file_claims", [])
            if file_claims:
                fnames = [os.path.basename(f) for f in file_claims[:3]]
                if len(file_claims) > 3:
                    fnames.append(f"+{len(file_claims) - 3}")
                coord_parts.append(f"Files: {', '.join(fnames)} ({len(file_claims)} released)")
            # Messages sent/received
            msg_sent = 0
            msg_received = 0
            try:
                sent_rows = mgr._conn.execute(
                    "SELECT COUNT(*) FROM coord_messages WHERE from_session = ?",
                    (session_id,),
                ).fetchone()
                msg_sent = sent_rows[0] if sent_rows else 0
                recv_rows = mgr._conn.execute(
                    "SELECT COUNT(*) FROM coord_messages WHERE to_session = ? OR "
                    "(to_session IS NULL AND from_session != ?)",
                    (session_id, session_id),
                ).fetchone()
                msg_received = recv_rows[0] if recv_rows else 0
            except Exception:
                pass
            if msg_sent or msg_received:
                coord_parts.append(f"Messages: {msg_sent} sent, {msg_received} received")
            # Tasks summary
            task_parts = []
            if my_completed:
                task_parts.append(f"{len(my_completed)} completed")
            if incomplete:
                for t in incomplete[:2]:
                    progress = t.get("progress", 0)
                    prog_str = f" ({progress}%)" if progress else ""
                    task_parts.append(f"#{t['id']} in progress{prog_str}")
            if task_parts:
                coord_parts.append(f"Tasks: {', '.join(task_parts)}")
            if coord_parts:
                lines.append("[COORD] Session coordination:")
                for part in coord_parts:
                    lines.append(f"  {part}")
        except Exception:
            pass  # Coord summary is best-effort

        mgr.deregister_session(session_id)

        # Footer: snapshot + peer notification
        footer = "Snapshot stored."
        if peer_count > 0:
            footer += f" {peer_count} peer(s) notified."
        lines.append(footer)

    except Exception as e:
        _log_hook_error("coord_session_stop", e)
        return {"output": "", "error": str(e)}

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
        # Bias query with session intent if available
        intent_hint = ""
        if session_id:
            cached_intent = _session_intent.get(session_id)
            _INTENT_HINTS = {
                "coding": "code implementation",
                "logic": "logic analysis",
                "creative": "creative writing",
                "exploration": "research exploration",
            }
            if cached_intent and cached_intent in _INTENT_HINTS:
                intent_hint = _INTENT_HINTS[cached_intent] + " "
        results = query_structured(
            query_text=f"{intent_hint}{filename} {dirname} {file_path}",
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

    # Surface relevant knowledge base chunks for this file
    if file_path:
        try:
            from omega.knowledge.engine import search_documents as _kb_search

            filename = os.path.basename(file_path)
            kb_result = _kb_search(query=filename, limit=1, entity_id=entity_id)
            if kb_result and "No document matches" not in kb_result and "No documents ingested" not in kb_result:
                # Extract first line of content (skip header formatting)
                kb_lines_raw = [ln.strip() for ln in kb_result.split("\n") if ln.strip() and not ln.startswith("#")]
                if kb_lines_raw:
                    snippet = kb_lines_raw[0][:120]
                    lines.append(f"[KB] {snippet}")
        except ImportError:
            pass  # Knowledge engine not available
        except Exception:
            pass  # KB surfacing is best-effort

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


def handle_coord_heartbeat(payload: dict) -> dict:
    """Update session heartbeat (debounced to max once per 30s).

    Every 5th heartbeat, check for unread messages and surface count.
    """
    session_id = payload.get("session_id", "")
    if not session_id:
        return {"output": "", "error": None}

    # Drain urgent queue BEFORE debounce — every tool use can deliver push notifications
    urgent_output = _drain_urgent_queue(session_id)

    now = time.monotonic()
    if now - _last_heartbeat.get(session_id, 0) < HEARTBEAT_DEBOUNCE_S:
        return {"output": urgent_output, "error": None}

    _last_heartbeat[session_id] = now
    output = urgent_output
    try:
        from omega.coordination import get_manager

        mgr = get_manager()
        mgr.heartbeat(session_id)

        # Single-agent fast path: skip coordination overhead when alone
        try:
            multi_agent = mgr.active_session_count() > 1
        except Exception:
            multi_agent = True  # Assume multi-agent when check fails

        # Every 2nd heartbeat, check inbox (~60s polling)
        # Non-4th beats: show inbox preview with nicknames (inform-type only,
        #   preserves request/complete unread for 4th-beat coordination check)
        # 4th beats: full coordination check below handles request/complete messages
        count = _heartbeat_count.get(session_id, 0) + 1
        _heartbeat_count[session_id] = count
        project = payload.get("project", "")

        # Mid-session activity pulse — fires once at heartbeat #5
        if count == 5:
            try:
                surfaced_json = Path.home() / ".omega" / f"session-{session_id}.surfaced.json"
                if surfaced_json.exists():
                    data = json.loads(surfaced_json.read_text())
                    surfaced_so_far = sum(len(ids) for ids in data.values())
                    if surfaced_so_far > 0:
                        output = _append_output(output, f"[MEMORY] {surfaced_so_far} memories surfaced so far this session")
            except Exception:
                pass  # Activity pulse is best-effort

        unread = 0
        if count % 2 == 0:
            try:
                unread = mgr.get_unread_count(session_id)
                if unread > 0 and count % 4 != 0:
                    # Show inbox preview with nicknames (inform-type only)
                    try:
                        msgs = mgr.check_inbox(session_id, unread_only=True, limit=2, msg_type="inform")
                        if msgs:
                            previews = []
                            for m in msgs[:2]:
                                from_name = _agent_nickname(m.get("from_session") or "unknown")
                                subj = (m.get("subject") or "")[:60]
                                previews.append(f'{from_name}: "{subj}"')
                            remaining = unread - len(msgs)
                            output = f"[INBOX] {unread} unread — " + " | ".join(previews)
                            if remaining > 0:
                                output += " — use omega_inbox for more"
                        else:
                            output = f"[INBOX] {unread} unread message(s) — use omega_inbox to read"
                    except Exception:
                        output = f"[INBOX] {unread} unread message(s) — use omega_inbox to read"
            except Exception:
                pass  # Inbox check is best-effort

        # Every 4th heartbeat (~2 min), surface messages + coordination check
        if count % 4 == 0:
            try:
                coord_lines = []

                # Multi-agent only: check if my task is blocked by another agent's task
                if multi_agent:
                    try:
                        all_tasks = mgr.list_tasks(project=project, status="in_progress")
                        my_tasks = [t for t in all_tasks if t.get("session_id") == session_id]
                        for task in my_tasks[:1]:
                            for dep_id in task.get("depends_on") or []:
                                dep_tasks = [t for t in all_tasks if t.get("id") == dep_id]
                                if dep_tasks and dep_tasks[0].get("status") != "completed":
                                    owner_name = _agent_nickname(dep_tasks[0].get("session_id") or "unknown")
                                    coord_lines.append(
                                        f"[BLOCKED] Task #{task['id']} waiting on #{dep_id} "
                                        f"(owner: {owner_name}, status: {dep_tasks[0].get('status', 'unknown')})"
                                    )
                    except Exception:
                        pass

                # Surface first request-type message content (not just count)
                if unread > 0:
                    try:
                        msgs = mgr.check_inbox(session_id, unread_only=True, limit=1, msg_type="request")
                        if msgs:
                            m = msgs[0]
                            from_name = _agent_nickname(m.get("from_session") or "unknown")
                            subj = (m.get("subject") or "")[:80]
                            coord_lines.append(f"[REQUEST] From {from_name}: {subj}")
                    except Exception:
                        pass

                # Surface first complete-type message (handoff from departed peer)
                try:
                    complete_msgs = mgr.check_inbox(session_id, unread_only=True, limit=1, msg_type="complete")
                    if complete_msgs:
                        m = complete_msgs[0]
                        from_name = _agent_nickname(m.get("from_session") or "unknown")
                        subj = (m.get("subject") or "")[:80]
                        body_preview = (m.get("body") or "")[:200].split("\n")[0]
                        coord_lines.append(
                            f"[HANDOFF] From {from_name}: {subj}" + (f"\n  {body_preview}" if body_preview else "")
                        )
                except Exception:
                    pass

                # Multi-agent only: [TEAM] activity line — recent peer coordination events
                if multi_agent and project:
                    try:
                        events = mgr.get_recent_events(
                            project=project,
                            minutes=2,
                            limit=3,
                            exclude_session=session_id,
                        )
                        if events:
                            parts = []
                            for ev in events[:2]:
                                ev_name = _agent_nickname(ev["session_id"])
                                parts.append(f"{ev_name} {ev['summary']}")
                            coord_lines.append(f"[TEAM] {' | '.join(parts)}")
                    except Exception:
                        pass  # Team activity is best-effort

                if coord_lines:
                    output = _append_output(output, "\n".join(coord_lines))
            except Exception:
                pass  # Coordination check is best-effort

        # Every 6th heartbeat (~3 min), idle detection — nudge agent to claim a task
        if count % 6 == 0:
            try:
                all_tasks = mgr.list_tasks(status="in_progress")
                my_tasks = [t for t in all_tasks if t.get("session_id") == session_id]
                if not my_tasks:
                    pending = mgr.list_tasks(project=project or None, status="pending")
                    unblocked = [t for t in pending if not t.get("blocked")]
                    if unblocked:
                        idle_line = (
                            f"[IDLE] No active task — {len(unblocked)} unclaimed task(s) available. "
                            f"Use omega_task_next to claim one."
                        )
                        output = _append_output(output, idle_line)
            except Exception:
                pass  # Idle detection is best-effort

        # Every 8th heartbeat (~4 min), detect peer state changes
        if multi_agent and count % 8 == 0 and project:
            try:
                sessions = mgr.list_sessions(auto_clean=False)
                current_peers = {
                    s["session_id"]
                    for s in sessions
                    if s.get("session_id") != session_id
                    and s.get("project") == project
                    and s.get("status") == "active"
                }
                prev_peers = _peer_snapshot.get(session_id, set())
                _peer_snapshot[session_id] = current_peers

                # Only report changes after first snapshot (skip initial population)
                if prev_peers is not None and prev_peers != current_peers:
                    change_parts = []
                    joined = current_peers - prev_peers
                    departed = prev_peers - current_peers
                    for sid in list(joined)[:2]:
                        change_parts.append(f"{_agent_nickname(sid)} joined")
                    for sid in list(departed)[:2]:
                        change_parts.append(f"{_agent_nickname(sid)} left")

                    # Check for new claims in same project from peers
                    for p_sid in list(current_peers)[:4]:
                        try:
                            claims = mgr.get_session_claims(p_sid)
                            new_files = claims.get("file_claims", [])
                            if new_files:
                                p_name = _agent_nickname(p_sid)
                                fname = os.path.basename(new_files[-1])
                                change_parts.append(f"{p_name} claimed {fname}")
                        except Exception:
                            pass

                    if change_parts:
                        active_count = len(current_peers)
                        summary = " | ".join(change_parts[:3])
                        refresh_line = f"[TEAM] {summary} | {active_count} peer{'s' if active_count != 1 else ''} active"
                        output = _append_output(output, refresh_line)
            except Exception:
                pass  # Peer refresh is best-effort

        # Every 10th heartbeat (~5 min), deadlock detection
        if multi_agent and count % 10 == 0:
            try:
                cycles = mgr.detect_deadlocks()
                if cycles:
                    now_dl = time.monotonic()
                    for cycle in cycles[:2]:
                        cycle_str = " -> ".join(_agent_nickname(s) for s in cycle)
                        deadlock_line = f"[DEADLOCK] Circular wait: {cycle_str}"
                        output = _append_output(output, deadlock_line)

                        # Push [DEADLOCK] to all OTHER sessions in the cycle (debounced)
                        cycle_key = str(hash(tuple(sorted(cycle[:-1]))))
                        if cycle_key not in _last_deadlock_push or now_dl - _last_deadlock_push[cycle_key] >= DEADLOCK_PUSH_DEBOUNCE_S:
                            _last_deadlock_push[cycle_key] = now_dl
                            for peer in set(cycle[:-1]):
                                if peer == session_id:
                                    continue  # detecting session sees it in hook output
                                try:
                                    mgr.send_message(
                                        from_session=session_id,
                                        subject=f"[DEADLOCK] Circular wait: {cycle_str}",
                                        to_session=peer,
                                        msg_type="inform",
                                        ttl_minutes=30,
                                    )
                                except Exception:
                                    pass  # fail-open per notification
            except Exception:
                pass  # Deadlock detection is best-effort

        # Every 20th heartbeat (~10 min), lightweight git divergence check
        if count % 20 == 0 and project:
            try:
                result = subprocess.run(
                    ["git", "rev-list", "--count", "--left-right", "HEAD...@{u}"],
                    capture_output=True, text=True, timeout=5, cwd=project,
                )
                if result.returncode == 0:
                    parts = result.stdout.strip().split("\t")
                    if len(parts) == 2:
                        ahead, behind = int(parts[0]), int(parts[1])
                        if behind > 0:
                            div_line = f"[GIT] Branch is {behind} commit(s) behind upstream"
                            if ahead > 0:
                                div_line += f" (and {ahead} ahead)"
                            output = _append_output(output, div_line)
            except Exception:
                pass  # Git divergence check is best-effort
    except Exception as e:
        _log_hook_error("coord_heartbeat", e)
        return {"output": "", "error": str(e)}

    return {"output": output, "error": None}


def _suggest_alternative_dir(file_path: str) -> str | None:
    """Suggest sibling directories as alternative work areas."""
    parent = os.path.dirname(file_path)
    grandparent = os.path.dirname(parent)
    if not grandparent or grandparent == parent:
        return None
    try:
        siblings = [
            d
            for d in sorted(os.listdir(grandparent))
            if os.path.isdir(os.path.join(grandparent, d))
            and d != os.path.basename(parent)
            and not d.startswith((".", "__"))
        ]
        if siblings:
            return f"nearby dirs: {', '.join(siblings[:3])}"
    except OSError:
        pass
    return None


def handle_auto_claim_file(payload: dict) -> dict:
    """Auto-claim files on Edit/Write/NotebookEdit (debounced per file)."""
    tool_name = payload.get("tool_name", "")
    if tool_name not in ("Edit", "Write", "NotebookEdit"):
        return {"output": "", "error": None}

    session_id = payload.get("session_id", "")
    if not session_id:
        return {"output": "", "error": None}

    input_data = _parse_tool_input(payload)
    file_path = _get_file_path_from_input(input_data)
    if not file_path:
        return {"output": "", "error": None}

    # Debounce: skip full claim if same (session, file) claimed recently,
    # but still refresh last_activity to prevent TTL expiry during active editing
    claim_key = (session_id, file_path)
    if not _debounce_check(_last_claim, claim_key, CLAIM_DEBOUNCE_S, _MAX_SURFACE_ENTRIES):
        try:
            from omega.coordination import get_manager

            mgr = get_manager()
            mgr.refresh_file_activity(session_id, file_path)
        except Exception:
            pass  # Activity refresh is best-effort
        return {"output": "", "error": None}

    output = ""
    try:
        from omega.coordination import get_manager

        mgr = get_manager()
        result = mgr.claim_file(session_id, file_path, task="auto-claimed on edit")
        if result.get("conflict"):
            owner_name = _agent_nickname(result["claimed_by"])
            owner_task = result.get("task") or "unknown task"
            output = (
                f"[CONFLICT] {os.path.basename(file_path)} is claimed by "
                f"{owner_name} ({owner_task}). Coordinate before editing."
            )
        elif result.get("success"):
            # Single-agent fast path: skip intent + overlap checks when alone
            try:
                multi_agent = mgr.active_session_count() > 1
            except Exception:
                multi_agent = True  # Assume multi-agent when check fails

            if multi_agent:
                # Auto-announce intent for coordination visibility
                try:
                    mgr.announce_intent(
                        session_id=session_id,
                        description=f"Editing {os.path.basename(file_path)}",
                        intent_type="edit",
                        target_files=[file_path],
                        ttl_minutes=5,
                    )
                except Exception:
                    pass  # Intent announcement is best-effort

                # Check for intent overlaps with other agents (best-effort, max 2 warnings)
                try:
                    overlap_result = mgr.check_intents(session_id)
                    if overlap_result.get("has_overlaps"):
                        overlaps = overlap_result["overlaps"][:2]
                        for ov in overlaps:
                            ov_sid_full = ov["session_id"]
                            ov_name = _agent_nickname(ov_sid_full)
                            ov_desc = ov.get("description", "")[:60]
                            ov_file_paths = ov.get("overlapping_files", [])[:3]
                            ov_files = ", ".join(os.path.basename(f) for f in ov_file_paths)

                            # Escalate if overlapping files are already CLAIMED by the other agent
                            escalated = False
                            try:
                                for of in ov_file_paths:
                                    check = mgr.check_file(of)
                                    if check.get("claimed") and check.get("session_id") == ov_sid_full:
                                        escalated = True
                                        break
                            except Exception:
                                pass  # Escalation check is best-effort

                            if escalated:
                                warning = (
                                    f"[CONFLICT] {ov_name} owns "
                                    f"{', '.join(os.path.basename(f) for f in ov_file_paths)}"
                                    f" — consider working in a different area"
                                )
                                alt = _suggest_alternative_dir(file_path)
                                if alt:
                                    warning += f"\n  Suggestion: {alt}"
                            else:
                                warning = f"[INTENT-OVERLAP] {ov_name}: {ov_desc}"
                                if ov_files:
                                    warning += f" (files: {ov_files})"

                            output += ("\n" if output else "") + warning

                            # Notify the other agent about the overlap (debounced)
                            try:
                                notify_key = (session_id, ov_sid_full, file_path)
                                notify_now = time.monotonic()
                                if (
                                    notify_key not in _last_overlap_notify
                                    or notify_now - _last_overlap_notify[notify_key] >= OVERLAP_NOTIFY_DEBOUNCE_S
                                ):
                                    _last_overlap_notify[notify_key] = notify_now
                                    filename = os.path.basename(file_path)
                                    mgr.send_message(
                                        from_session=session_id,
                                        to_session=ov_sid_full,
                                        msg_type="inform",
                                        subject=f"Overlap: both editing {filename}",
                                        body=(
                                            f"I'm editing {file_path}. You announced intent "
                                            f"to work on overlapping files. Let's coordinate."
                                        ),
                                        ttl_minutes=30,
                                    )
                            except Exception:
                                pass  # Overlap notification is best-effort
                except Exception:
                    pass  # Overlap check is best-effort
    except Exception as e:
        error_str = str(e)
        if "already claimed" not in error_str.lower():
            _log_hook_error("auto_claim_file", e)

    return {"output": output, "error": None}


def _file_guard_block_msg(file_path: str, owner_sid: str, owner_task: str, blocked_sid: str = "") -> dict:
    """Return a block response for the file guard.

    Also sends a [WAITING] message to the file owner so they know
    someone is blocked (debounced to once per 5 min per tuple).
    """
    filename = os.path.basename(file_path)
    owner_name = _agent_nickname(owner_sid)
    msg = (
        f"\n[FILE-GUARD] BLOCKED: {filename} is claimed by {owner_name} ({owner_task}).\n"
        f"  Options:\n"
        f"    1. Wait for the other agent to finish and release\n"
        f"    2. Ask other agent to call omega_file_release\n"
        f"    3. The claim expires automatically after 10 minutes of inactivity\n"
        f"    4. Ask the human to decide if a force-override is safe"
    )

    # Notify the owner that someone is waiting
    if blocked_sid and owner_sid:
        now = time.monotonic()
        key = (blocked_sid, owner_sid, file_path)
        if key not in _last_block_notify or now - _last_block_notify[key] >= BLOCK_NOTIFY_DEBOUNCE_S:
            _last_block_notify[key] = now
            try:
                from omega.coordination import get_manager

                mgr = get_manager()
                blocked_name = _agent_nickname(blocked_sid)
                mgr.send_message(
                    from_session=blocked_sid,
                    subject=f"[WAITING] {blocked_name} wants to edit {filename} — consider releasing if done",
                    msg_type="inform",
                    to_session=owner_sid,
                )
            except Exception:
                pass  # Fail-open — notification is best-effort

    return {"output": msg, "error": None, "exit_code": 2}


def handle_pre_file_guard(payload: dict) -> dict:
    """Check file claims BEFORE editing — blocks if claimed by another agent.

    Returns exit_code=2 in the response dict when blocking.
    Fail-open: any error returns allow (exit_code=0).
    """
    tool_name = payload.get("tool_name", "")
    if tool_name not in ("Edit", "Write", "NotebookEdit"):
        return {"output": "", "error": None}

    session_id = payload.get("session_id", "")
    input_data = _parse_tool_input(payload)
    file_path = _get_file_path_from_input(input_data)
    if not file_path:
        return {"output": "", "error": None}

    try:
        from omega.coordination import get_manager

        mgr = get_manager()
        info = mgr.check_file(file_path)

        if info.get("claimed"):
            if session_id and info.get("session_id") == session_id:
                return {"output": "", "error": None}

            # Claimed by different session (or no session_id to prove identity) — BLOCK
            owner_sid = info.get("session_id", "unknown")
            owner_task = info.get("task") or "unknown task"
            return _file_guard_block_msg(file_path, owner_sid, owner_task, blocked_sid=session_id)

        # Unclaimed — if we have session_id, claim atomically to prevent TOCTOU race
        if session_id:
            result = mgr.claim_file(session_id, file_path, task="pre-edit guard claim")
            if result.get("conflict"):
                owner_sid = result["claimed_by"]
                owner_task = result.get("task") or "unknown task"
                return _file_guard_block_msg(file_path, owner_sid, owner_task, blocked_sid=session_id)

        # No session_id + unclaimed → allow (true single-agent)
        return {"output": "", "error": None}

    except Exception as e:
        # Fail-open: never block when OMEGA is unavailable
        _log_hook_error("pre_file_guard", e)
        return {"output": "", "error": None}


def handle_pre_task_guard(payload: dict) -> dict:
    """Check task declaration BEFORE editing — blocks if no active task.

    Opt-in: only enforces when the project has non-terminal tasks.
    Returns exit_code=2 in the response dict when blocking.
    Fail-open: any error returns allow (exit_code absent).
    """
    tool_name = payload.get("tool_name", "")
    if tool_name not in ("Edit", "Write", "NotebookEdit"):
        return {"output": "", "error": None}

    session_id = payload.get("session_id", "")
    if not session_id:
        # Single-agent mode — no enforcement
        return {"output": "", "error": None}

    input_data = _parse_tool_input(payload)
    file_path = _get_file_path_from_input(input_data)
    if not file_path:
        return {"output": "", "error": None}

    project = payload.get("project", "")
    if not project:
        return {"output": "", "error": None}

    # Skip if file is outside the project directory
    try:
        if not os.path.abspath(file_path).startswith(os.path.abspath(project)):
            return {"output": "", "error": None}
    except Exception:
        return {"output": "", "error": None}

    try:
        from omega.coordination import get_manager

        mgr = get_manager()

        # Opt-in: only enforce if project has active tasks
        if not mgr.project_has_active_tasks(project):
            return {"output": "", "error": None}

        # Check if session has an in_progress task
        result = mgr.has_active_task(session_id)
        if result.get("has_task"):
            return {"output": "", "error": None}

        # No active task — BLOCK
        filename = os.path.basename(file_path)
        project_name = os.path.basename(project)
        msg = (
            f"\n[TASK-GUARD] BLOCKED: No active task for this session on {project_name}.\n"
            f"  Create and claim a task before editing {filename}:\n"
            f'    1. omega_task_create(title="Your task", project="{project}")\n'
            f'    2. omega_task_claim(task_id=<id>, session_id="{session_id}")\n'
            f"  Or complete/cancel existing tasks to disable enforcement."
        )
        return {"output": msg, "error": None, "exit_code": 2}

    except Exception as e:
        # Fail-open: never block when OMEGA is unavailable
        _log_hook_error("pre_task_guard", e)
        return {"output": "", "error": None}


def _clean_task_text(prompt: str) -> str:
    """Delegate to shared implementation in omega.task_utils."""
    from omega.task_utils import clean_task_text
    return clean_task_text(prompt)


def handle_auto_capture(payload: dict) -> dict:
    """Auto-capture decisions and lessons from user prompts (UserPromptSubmit)."""
    # Prefer top-level keys (set by fast_hook.py from parsed stdin JSON).
    # Fall back to re-parsing payload["stdin"] for legacy/direct callers.
    prompt = payload.get("prompt", "")
    stdin_parsed = {}
    if not prompt:
        stdin_data = payload.get("stdin", "")
        if not stdin_data:
            return {"output": "", "error": None}
        try:
            stdin_parsed = json.loads(stdin_data)
        except (json.JSONDecodeError, TypeError):
            return {"output": "", "error": None}
        prompt = stdin_parsed.get("prompt", "")

    # Extract session_id/cwd from top-level first, fall back to parsed stdin
    session_id = payload.get("session_id") or stdin_parsed.get("session_id", "")
    cwd = payload.get("cwd") or payload.get("project") or stdin_parsed.get("cwd", "")
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
            if session_id:
                _session_intent[session_id] = intent
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


# ---------------------------------------------------------------------------
# Pre-push guard (git divergence + branch claims)
# ---------------------------------------------------------------------------


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


def _parse_checkout_target(command: str) -> str | None:
    """Parse the target branch from git checkout/switch commands.

    Returns None for new branch creation (-b/-B/-c/-C/--orphan)
    and file restores (-- separator).
    """
    for segment in re.split(r"&&|\|\||;", command):
        segment = segment.strip()
        match = re.match(r"git\s+(checkout|switch)\s+(.*)", segment)
        if not match:
            continue
        args_str = match.group(2).strip()

        if re.search(r"(?:^|\s)-[bBcC]\b", args_str):
            return None
        if "--orphan" in args_str:
            return None
        if " -- " in args_str or args_str.startswith("-- "):
            return None

        for token in args_str.split():
            if token.startswith("-"):
                continue
            return token

    return None


def handle_pre_push_guard(payload: dict) -> dict:
    """Git push divergence guard + branch claim check.

    Enforces:
      1. git push: blocks if origin has advanced (divergence guard)
      2. git checkout/switch: blocks if target branch is claimed by another agent
      3. git commit: blocks if current branch is claimed by another agent

    Returns exit_code=2 when blocking. Fail-open on errors.
    """
    tool_name = payload.get("tool_name", "")
    if tool_name != "Bash":
        return {"output": "", "error": None}

    input_data = _parse_tool_input(payload)
    command = input_data.get("command", "")
    session_id = payload.get("session_id", "")
    project = payload.get("project", "")

    if re.search(r"\bgit\s+push\b", command):
        result = _handle_push_divergence(command, project, session_id)
        if result.get("exit_code"):
            return result
        _handle_auto_claim_branch(command, session_id, project)
        return {"output": "", "error": None}

    if re.search(r"\bgit\s+(?:checkout|switch|commit)\b", command):
        return _handle_branch_claims(command, session_id, project)

    return {"output": "", "error": None}


def _handle_push_divergence(command: str, project: str, session_id: str) -> dict:
    """Check for push divergence. Returns exit_code=2 if origin has advanced."""
    if not project:
        return {"output": "", "error": None}

    try:
        result = subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=project,
        )
        if result.returncode != 0:
            return {"output": "", "error": None}

        subprocess.run(
            ["git", "fetch", "origin", "--quiet"],
            capture_output=True,
            text=True,
            timeout=15,
            cwd=project,
        )

        branch = _get_current_branch(project) or "main"

        behind_result = subprocess.run(
            ["git", "log", f"HEAD..origin/{branch}", "--oneline", "--no-decorate", "-10"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=project,
        )
        if behind_result.returncode != 0 or not behind_result.stdout.strip():
            # Not behind — log push event
            _log_push_event(project, branch, session_id)
            return {"output": "", "error": None}

        behind_lines = behind_result.stdout.strip().split("\n")
        count = len(behind_lines)

        # Log divergence to coordination
        try:
            from omega.coordination import get_manager

            mgr = get_manager()
            mgr.log_git_event(
                project=project,
                event_type="push_divergence_warning",
                branch=branch,
                message=f"{count} upstream commit(s) detected before push",
                session_id=session_id,
            )
        except Exception:
            pass

        lines = [f"\n[GIT-GUARD] BLOCKED: origin/{branch} has {count} commit(s) not in HEAD:"]
        for line in behind_lines[:5]:
            lines.append(f"  {line}")
        if count > 5:
            lines.append(f"  ... and {count - 5} more")
        lines.append("  Run 'git pull --rebase' before pushing to avoid conflicts.")

        return {"output": "\n".join(lines), "error": None, "exit_code": 2}

    except (subprocess.TimeoutExpired, FileNotFoundError):
        return {"output": "", "error": None}
    except Exception as e:
        _log_hook_error("pre_push_guard", e)
        return {"output": "", "error": None}


def _handle_branch_claims(command: str, session_id: str, project: str) -> dict:
    """Check branch claims for checkout/switch/commit commands."""
    if not session_id:
        return {"output": "", "error": None}

    try:
        if "git commit" in command:
            branch = _get_current_branch(project)
            if branch:
                return _block_if_branch_claimed(session_id, project, branch)
            return {"output": "", "error": None}

        target = _parse_checkout_target(command)
        if target:
            return _block_if_branch_claimed(session_id, project, target)

    except Exception as e:
        _log_hook_error("branch_guard", e)

    return {"output": "", "error": None}


def _block_if_branch_claimed(session_id: str, project: str, branch: str) -> dict:
    """Block if the branch is claimed by another agent."""
    try:
        from omega.coordination import get_manager

        mgr = get_manager()
        info = mgr.check_branch(project, branch)

        if not info.get("claimed"):
            return {"output": "", "error": None}

        if info.get("session_id") == session_id:
            return {"output": "", "error": None}  # Self-claim

        owner_name = _agent_nickname(info.get("session_id", "unknown"))
        owner_task = info.get("task") or "unknown task"
        msg = (
            f"\n[BRANCH-GUARD] BLOCKED: branch '{branch}' is claimed by {owner_name} ({owner_task}).\n"
            f"  Options:\n"
            f"    1. Wait for the other agent to finish\n"
            f"    2. Ask other agent to call omega_branch_release\n"
            f"    3. Use a different feature branch"
        )
        return {"output": msg, "error": None, "exit_code": 2}

    except ImportError:
        return {"output": "", "error": None}
    except Exception as e:
        _log_hook_error("branch_guard", e)
        return {"output": "", "error": None}


def _handle_auto_claim_branch(command: str, session_id: str, project: str):
    """Auto-claim the current branch before a push succeeds."""
    if not session_id or not project or not os.path.isdir(project):
        return
    try:
        branch = _get_current_branch(project)
        if not branch or branch == "HEAD":
            return
        from omega.coordination import get_manager

        mgr = get_manager()
        mgr.claim_branch(project=project, branch=branch, session_id=session_id, task="pushing to remote")
    except Exception:
        pass


def _log_push_event(project: str, branch: str, session_id: str):
    """Log a push event to coordination."""
    try:
        from omega.coordination import get_manager

        mgr = get_manager()

        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=project,
        )
        commit_hash = result.stdout.strip() if result.returncode == 0 else None

        mgr.log_git_event(
            project=project,
            event_type="push",
            commit_hash=commit_hash,
            branch=branch,
            session_id=session_id,
        )
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Handler dispatch table — core handlers are always available; additional
# handlers (coordination, etc.) are loaded when present or provided by plugins.
# ---------------------------------------------------------------------------

# Core memory handlers — always shipped with omega-memory
_CORE_HOOK_HANDLERS = {
    "session_start": handle_session_start,
    "session_stop": handle_session_stop,
    "surface_memories": handle_surface_memories,
    "auto_capture": handle_auto_capture,
}

# Additional handlers — loaded when available, or provided by plugins
_EXTENDED_HOOK_HANDLERS = {
    "coord_session_start": handle_coord_session_start,
    "coord_session_stop": handle_coord_session_stop,
    "coord_heartbeat": handle_coord_heartbeat,
    "auto_claim_file": handle_auto_claim_file,
    "pre_file_guard": handle_pre_file_guard,
    "pre_task_guard": handle_pre_task_guard,
    "pre_push_guard": handle_pre_push_guard,
}

# Build the dispatch table: core + extended (if available) + plugins
HOOK_HANDLERS = dict(_CORE_HOOK_HANDLERS)

# Coordination handlers: only register if the coordination module is available.
# In core-only installs (PyPI omega), these remain defined but unreachable —
# hooks-core.json won't wire them, and this guard prevents accidental dispatch.
try:
    import omega.coordination  # noqa: F401
    HOOK_HANDLERS.update(_EXTENDED_HOOK_HANDLERS)
except ImportError:
    pass

# Discover plugin-provided hook handlers (e.g. omega-pro adds its own hooks)
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
