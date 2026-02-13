#!/usr/bin/env python3
"""OMEGA SessionStop hook — Generate and store session summary on exit."""
import json
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path


def _log_hook_error(hook_name, error):
    try:
        log_path = Path.home() / ".omega" / "hooks.log"
        log_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        timestamp = datetime.now().isoformat(timespec="seconds")
        tb = traceback.format_exc()
        data = f"[{timestamp}] {hook_name}: {error}\n{tb}\n"
        fd = os.open(str(log_path), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            os.write(fd, data.encode("utf-8"))
        finally:
            os.close(fd)
    except Exception:
        pass


def _get_activity_counts(session_id: str) -> dict:
    """Count memories by event_type for this session."""
    try:
        from omega.bridge import _get_store
        store = _get_store()
        return store.get_session_event_counts(session_id)
    except Exception:
        return {}


def _get_surfaced_count(session_id: str) -> int:
    """Read and clean up the surfacing counter file."""
    try:
        marker = Path.home() / ".omega" / f"session-{session_id}.surfaced"
        if marker.exists():
            count = marker.stat().st_size
            marker.unlink()
            return count
    except Exception:
        pass
    return 0


def _get_surfaced_details(session_id: str) -> tuple:
    """Read unique memory IDs and file count from surfaced.json."""
    unique_ids = 0
    unique_files = 0
    try:
        json_path = Path.home() / ".omega" / f"session-{session_id}.surfaced.json"
        if json_path.exists():
            data = json.loads(json_path.read_text())
            all_ids = set()
            for ids in data.values():
                all_ids.update(ids)
            unique_ids = len(all_ids)
            unique_files = len(data)
    except Exception:
        pass
    return unique_ids, unique_files


def _print_activity_report(session_id: str):
    """Print session memory activity summary with productivity recap."""
    if not session_id:
        return
    counts = _get_activity_counts(session_id)
    surfaced = _get_surfaced_count(session_id)
    surfaced_unique_ids, surfaced_unique_files = _get_surfaced_details(session_id)
    if not counts and surfaced == 0:
        return

    captured = sum(counts.values())
    parts = [f"{captured} captured"]
    _LABELS = {
        "error_pattern": ("error", "errors"),
        "decision": ("decision", "decisions"),
        "lesson_learned": ("lesson learned", "lessons learned"),
    }
    for key, (singular, plural) in _LABELS.items():
        n = counts.get(key, 0)
        if n:
            parts.append(f"{n} {plural if n > 1 else singular}")
    if surfaced:
        parts.append(f"{surfaced} surfaced")
    print(f"\n## Session complete — {' | '.join(parts)}")

    # Unique recall stats
    if surfaced_unique_ids > 0:
        print(f"  Recalled: {surfaced_unique_ids} unique memories across {surfaced_unique_files} file{'s' if surfaced_unique_files != 1 else ''}")

    # Weekly recap
    try:
        from omega.bridge import _get_store
        store = _get_store()
        total = store.node_count()

        from datetime import timedelta, timezone
        week_cutoff = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
        row = store._conn.execute(
            "SELECT COUNT(DISTINCT session_id) FROM memories "
            "WHERE created_at >= ? AND session_id IS NOT NULL",
            (week_cutoff,),
        ).fetchone()
        weekly_sessions = row[0] if row else 0

        row2 = store._conn.execute(
            "SELECT COUNT(*) FROM memories WHERE created_at >= ?",
            (week_cutoff,),
        ).fetchone()
        weekly_memories = row2[0] if row2 else 0

        # Prior week count for growth
        prev_cutoff = (datetime.now(timezone.utc) - timedelta(days=14)).isoformat()
        row3 = store._conn.execute(
            "SELECT COUNT(*) FROM memories WHERE created_at >= ? AND created_at < ?",
            (prev_cutoff, week_cutoff),
        ).fetchone()
        prev_week_memories = row3[0] if row3 else 0

        recap_parts = []
        if weekly_sessions > 1:
            recap_parts.append(f"{weekly_sessions} sessions this week")
        if weekly_memories > 0:
            recap_parts.append(f"{weekly_memories} memories this week")
        recap_parts.append(f"{total} total")
        print(f"  Recap: {', '.join(recap_parts)}")

        # Week-over-week growth
        if prev_week_memories > 0 and weekly_memories > 0:
            growth_pct = ((weekly_memories - prev_week_memories) / prev_week_memories) * 100
            sign = "+" if growth_pct >= 0 else ""
            print(f"  Growth: {sign}{growth_pct:.0f}% vs last week")
    except Exception:
        pass


def _build_summary(session_id: str, project: str) -> str:
    """Build a session summary from per-type targeted queries.

    Each category is queried independently with event_type filter.
    session_summary type is excluded entirely to prevent circular refs.
    """
    try:
        from omega.bridge import query_structured
    except ImportError:
        return "Session ended"

    decisions = query_structured(
        query_text="decisions made",
        limit=5,
        session_id=session_id,
        project=project,
        event_type="decision",
    )
    errors = query_structured(
        query_text="errors encountered",
        limit=3,
        session_id=session_id,
        project=project,
        event_type="error_pattern",
    )
    tasks = query_structured(
        query_text="completed tasks",
        limit=3,
        session_id=session_id,
        project=project,
        event_type="task_completion",
    )

    if not decisions and not errors and not tasks:
        return "Session ended (no captured activity)"

    parts = []
    if decisions:
        items = [m.get("content", "")[:120] for m in decisions[:3]]
        parts.append(f"Decisions ({len(decisions)}): " + "; ".join(items))
    if errors:
        items = [m.get("content", "")[:120] for m in errors[:3]]
        parts.append(f"Errors ({len(errors)}): " + "; ".join(items))
    if tasks:
        items = [m.get("content", "")[:120] for m in tasks[:3]]
        parts.append(f"Tasks ({len(tasks)}): " + "; ".join(items))

    if not parts:
        return "Session ended"

    return " | ".join(parts)[:600]


def _auto_feedback_on_surfaced(session_id: str):
    """Auto-record 'helpful' feedback for memories surfaced during active work."""
    if not session_id:
        return
    json_path = Path.home() / ".omega" / f"session-{session_id}.surfaced.json"
    if not json_path.exists():
        return
    try:
        data = json.loads(json_path.read_text())
        # Collect all unique memory IDs across all files
        all_ids = set()
        for ids in data.values():
            all_ids.update(ids)

        if not all_ids:
            return

        from omega.bridge import record_feedback
        count = 0
        for mid in list(all_ids)[:10]:  # Cap at 10 feedback calls
            try:
                record_feedback(mid, "helpful", "Auto: surfaced during active work")
                count += 1
            except Exception:
                pass

        # Clean up the JSON file
        json_path.unlink(missing_ok=True)
    except ImportError:
        pass
    except Exception as e:
        _log_hook_error("auto_feedback_surfaced", e)
    finally:
        # Always try to clean up
        try:
            if json_path.exists():
                json_path.unlink()
        except Exception:
            pass


def main():
    session_id = os.environ.get("SESSION_ID", "")
    project = os.environ.get("PROJECT_DIR", os.getcwd())

    _auto_feedback_on_surfaced(session_id)
    _print_activity_report(session_id)

    summary = _build_summary(session_id, project)

    try:
        from omega.bridge import auto_capture
        auto_capture(
            content=f"Session summary: {summary}",
            event_type="session_summary",
            metadata={"source": "session_stop_hook", "project": project},
            session_id=session_id,
            project=project,
        )
    except ImportError:
        pass
    except Exception as e:
        _log_hook_error("session_stop", e)
        print(f"OMEGA session_stop failed: {e}", file=sys.stderr)


def _log_timing(hook_name, elapsed_ms):
    try:
        log_path = Path.home() / ".omega" / "hooks.log"
        log_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        timestamp = datetime.now().isoformat(timespec="seconds")
        data = f"[{timestamp}] {hook_name}: OK ({elapsed_ms:.0f}ms)\n"
        fd = os.open(str(log_path), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            os.write(fd, data.encode("utf-8"))
        finally:
            os.close(fd)
    except Exception:
        pass


if __name__ == "__main__":
    _t0 = time.monotonic()
    main()
    _log_timing("session_stop", (time.monotonic() - _t0) * 1000)
