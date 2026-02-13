#!/usr/bin/env python3
"""OMEGA SessionStart hook — Welcome briefing with recent context."""
import os
import time
import traceback
from datetime import datetime, timezone
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


def _maybe_auto_consolidate():
    """Run lightweight consolidation if >7 days since last run."""
    try:
        marker = Path.home() / ".omega" / "last-consolidate"
        if marker.exists():
            last_ts = marker.read_text().strip()
            last = datetime.fromisoformat(last_ts)
            if last.tzinfo is None:
                last = last.replace(tzinfo=timezone.utc)
            age_days = (datetime.now(timezone.utc) - last).days
            if age_days < 7:
                return
        from omega.bridge import consolidate
        consolidate(prune_days=30, max_summaries=50)
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(datetime.now(timezone.utc).isoformat())
    except ImportError:
        pass
    except Exception as e:
        _log_hook_error("auto_consolidate", e)


def _maybe_auto_backup():
    """Export a backup if >7 days since last backup. Keep last 4."""
    try:
        marker = Path.home() / ".omega" / "last-backup"
        if marker.exists():
            last_ts = marker.read_text().strip()
            last = datetime.fromisoformat(last_ts)
            if last.tzinfo is None:
                last = last.replace(tzinfo=timezone.utc)
            age_days = (datetime.now(timezone.utc) - last).days
            if age_days < 7:
                return
        backup_dir = Path.home() / ".omega" / "backups"
        backup_dir.mkdir(parents=True, exist_ok=True)
        dest = backup_dir / f"omega-{datetime.now(timezone.utc).strftime('%Y-%m-%d')}.json"
        from omega.bridge import export_memories
        export_memories(filepath=str(dest))
        # Rotate: keep only last 4
        backups = sorted(backup_dir.glob("omega-*.json"), key=lambda p: p.name, reverse=True)
        for old in backups[4:]:
            old.unlink()
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(datetime.now(timezone.utc).isoformat())
    except ImportError:
        pass
    except Exception as e:
        _log_hook_error("auto_backup", e)


def _maybe_auto_compact():
    """Run compaction of lesson_learned memories if >14 days since last run."""
    try:
        marker = Path.home() / ".omega" / "last-compact"
        if marker.exists():
            last_ts = marker.read_text().strip()
            last = datetime.fromisoformat(last_ts)
            if last.tzinfo is None:
                last = last.replace(tzinfo=timezone.utc)
            age_days = (datetime.now(timezone.utc) - last).days
            if age_days < 14:
                return
        from omega.bridge import compact
        compact(event_type="lesson_learned", similarity_threshold=0.60, min_cluster_size=3)
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(datetime.now(timezone.utc).isoformat())
    except ImportError:
        pass
    except Exception as e:
        _log_hook_error("auto_compact", e)


def main():
    project = os.environ.get("PROJECT_DIR", os.getcwd())
    session_id = os.environ.get("SESSION_ID", "")

    # Auto-consolidation check (lightweight, max once per 7 days)
    _maybe_auto_consolidate()

    # Auto-compaction check (max once per 14 days)
    _maybe_auto_compact()

    # Auto-backup check (max once per 7 days)
    _maybe_auto_backup()

    try:
        from omega.bridge import welcome
        result = welcome(session_id=session_id, project=project)
    except ImportError:
        print("OMEGA not installed. Run: pip install omega-memory && omega setup")
        return
    except Exception as e:
        _log_hook_error("session_start", e)
        print(f"OMEGA welcome failed: {e}")
        return

    greeting = result.get("greeting", "")
    memory_count = result.get("memory_count", 0)
    recent = result.get("recent_memories", [])

    print(f"## {greeting}")

    # First-time user "Aha" moment
    if memory_count == 0:
        print("")
        print("OMEGA captures decisions, lessons, and errors automatically as you work.")
        print("Next session, it surfaces relevant context when you edit the same files.")
        print("")
        print("**Quick start:**")
        print('- Say "remember that we always use TypeScript strict mode" to store a preference')
        print("- Make a decision and OMEGA captures it automatically")
        print("- Encounter an error, and OMEGA stores the pattern for future recall")
        print("")
        print("After this session ends, you'll see exactly what was captured.")
    elif memory_count <= 10:
        print(f"  OMEGA has {memory_count} memories from your first sessions. These will surface when you edit related files.")
        try:
            from omega.bridge import type_stats as _ts_first
            first_stats = _ts_first()
            stat_parts = []
            for k, v in sorted(first_stats.items(), key=lambda x: x[1], reverse=True):
                if v > 0 and k != "session_summary":
                    stat_parts.append(f"{v} {k.replace('_', ' ')}")
            if stat_parts:
                print(f"  Captured so far: {', '.join(stat_parts[:4])}")
        except Exception:
            pass

    # Health pulse
    try:
        from datetime import timezone
        from omega.bridge import _get_store, status as omega_status
        health = omega_status()
        health_label = "ok" if health.get("ok") else health.get("status", "unknown")

        store = _get_store()
        edge_count = store.edge_count()
        last_ts = store.get_last_capture_time()
        if last_ts:
            last_dt = datetime.fromisoformat(last_ts.replace("Z", "+00:00"))
            if last_dt.tzinfo is None:
                last_dt = last_dt.replace(tzinfo=timezone.utc)
            delta = datetime.now(timezone.utc) - last_dt
            secs = delta.total_seconds()
            ago = (f"{int(secs)}s ago" if secs < 60
                   else f"{int(secs/60)}m ago" if secs < 3600
                   else f"{int(secs/3600)}h ago" if secs < 86400
                   else f"{int(secs/86400)}d ago")
        else:
            ago = "never"
        node_count = store.count()
        if node_count > 0:
            ratio = edge_count / node_count
            graph_label = "rich" if ratio >= 1.5 else ("good" if ratio >= 0.5 else "sparse")
            graph_info = f" | **Graph:** {graph_label} ({edge_count:,} edges, {ratio:.1f}x)"
        else:
            graph_info = ""
        print(f"**Health:** {health_label} | **Last capture:** {ago}{graph_info}")
    except Exception:
        pass

    # Profile summary
    try:
        from omega.profile.engine import get_profile_engine
        _pe = get_profile_engine()
        with _pe._lock:
            _prow = _pe._conn.execute("SELECT COUNT(*) as cnt FROM secure_profile").fetchone()
        _pcnt = _prow["cnt"] if _prow else 0
        if _pcnt > 0:
            print(f"**Profile:** {_pcnt} encrypted field(s) stored")
    except Exception:
        pass

    # Proactive maintenance suggestion when graph is sparse
    try:
        if node_count > 0 and edge_count / node_count < 0.5:
            print("[MAINTENANCE] Graph connectivity is sparse — consider running omega_compact to consolidate related memories")
    except Exception:
        pass

    # Type stats — show top-3 memory types
    try:
        from omega.bridge import type_stats
        stats = type_stats()
        if stats:
            top3 = sorted(stats.items(), key=lambda x: x[1], reverse=True)[:3]
            parts = [f"{k}: {v}" for k, v in top3]
            print(f"**Store:** {' | '.join(parts)}")
    except Exception:
        pass

    # Preferences count
    try:
        from omega.bridge import list_preferences
        prefs = list_preferences()
        if prefs:
            print(f"**Preferences:** {len(prefs)} stored")
    except Exception:
        pass

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

    # Cross-project lesson surfacing
    try:
        from omega.bridge import get_cross_project_lessons
        cross_lessons = get_cross_project_lessons(
            task=None,
            exclude_project=project,
            limit=3,
        )
        cross_only = [l for l in cross_lessons if l.get("cross_project")]
        if cross_only:
            print("\n[CROSS-PROJECT] Lessons from other codebases:")
            for l in cross_only[:3]:
                content = l.get("content", "")[:120]
                source_proj = l.get("project", "unknown")
                print(f"  - [{source_proj}] {content}")
    except ImportError:
        pass
    except Exception as e:
        _log_hook_error("cross_project_lessons", e)

    # Surface top project lessons
    try:
        from omega.bridge import get_cross_session_lessons
        project_lessons = get_cross_session_lessons(
            task=None,
            project_path=project,
            exclude_session=session_id,
            limit=3,
        )
        top_lessons = [l for l in project_lessons if (l.get("access_count", 0) or 0) > 0]
        if top_lessons:
            print("\n[LESSONS] Top lessons for this project:")
            for l in top_lessons[:3]:
                content = l.get("content", "")[:120]
                print(f"  - {content}")
    except ImportError:
        pass
    except Exception as e:
        _log_hook_error("project_lessons", e)

    # Auto-surfaced weekly digest (max once per 7 days, 20+ memories)
    if memory_count >= 20:
        try:
            marker = Path.home() / ".omega" / "last-digest"
            should_digest = True
            if marker.exists():
                last_ts = marker.read_text().strip()
                last = datetime.fromisoformat(last_ts)
                if last.tzinfo is None:
                    last = last.replace(tzinfo=timezone.utc)
                age_days = (datetime.now(timezone.utc) - last).days
                if age_days < 7:
                    should_digest = False

            if should_digest:
                from omega.bridge import get_weekly_digest
                digest = get_weekly_digest(days=7)
                period_new = digest.get("period_new", 0)
                session_count = digest.get("session_count", 0)
                total = digest.get("total_memories", 0)
                growth_pct = digest.get("growth_pct", 0)
                type_breakdown = digest.get("type_breakdown", {})

                if period_new > 0:
                    print(f"\n[WEEKLY] This week: {period_new} memories across {session_count} sessions")
                    if type_breakdown:
                        bd_parts = [f"{v} {k.replace('_', ' ')}" for k, v in sorted(type_breakdown.items(), key=lambda x: x[1], reverse=True)[:3]]
                        print(f"  Breakdown: {', '.join(bd_parts)}")
                    sign = "+" if growth_pct >= 0 else ""
                    print(f"  Trend: {sign}{growth_pct:.0f}% vs prior week | {total} total memories")

                    marker.parent.mkdir(parents=True, exist_ok=True)
                    marker.write_text(datetime.now(timezone.utc).isoformat())
        except ImportError:
            pass
        except Exception as e:
            _log_hook_error("weekly_digest", e)

    if recent:
        print(f"\n**Recent memories ({memory_count} total):**")
        for mem in recent[:3]:
            content = mem.get("content", "")[:100]
            print(f"- {content}")


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
    _log_timing("session_start", (time.monotonic() - _t0) * 1000)
