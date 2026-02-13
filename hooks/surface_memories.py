#!/usr/bin/env python3
"""OMEGA PostToolUse hook — Semantic memory surfacing and error auto-capture.

Triggered on Edit/Write/NotebookEdit/Bash. Provides:
- Semantic search for memories related to files being edited
- Auto-capture of error patterns from Bash failures
- Post-commit tracking for git-aware coordination
"""
import json
import os
import re
import time
import traceback
from datetime import datetime
from pathlib import Path


_MAX_LOG_BYTES = 5 * 1024 * 1024  # 5 MB cap


def _check_milestone(name: str) -> bool:
    """Return True if milestone not yet achieved (first time). Creates marker."""
    marker = Path.home() / ".omega" / "milestones" / name
    if marker.exists():
        return False
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.touch()
    return True


def _relative_time_from_iso(iso_str: str) -> str:
    """Convert an ISO timestamp to a human-readable relative time like '2d ago'."""
    try:
        dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            from datetime import timezone
            dt = dt.replace(tzinfo=timezone.utc)
        from datetime import timezone
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


def _rotate_log_if_needed(log_path: Path):
    """Rotate hooks.log if it exceeds the size cap."""
    try:
        if log_path.exists() and log_path.stat().st_size > _MAX_LOG_BYTES:
            rotated = log_path.with_suffix(".log.1")
            if rotated.exists():
                rotated.unlink()
            log_path.rename(rotated)
    except Exception:
        pass


def _log_hook_error(hook_name: str, error: Exception):
    """Log hook errors to ~/.omega/hooks.log for debugging."""
    try:
        log_path = Path.home() / ".omega" / "hooks.log"
        log_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        _rotate_log_if_needed(log_path)
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


def _log_timing(hook_name: str, elapsed_ms: float):
    try:
        log_path = Path.home() / ".omega" / "hooks.log"
        log_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        _rotate_log_if_needed(log_path)
        timestamp = datetime.now().isoformat(timespec="seconds")
        data = f"[{timestamp}] {hook_name}: OK ({elapsed_ms:.0f}ms)\n"
        fd = os.open(str(log_path), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            os.write(fd, data.encode("utf-8"))
        finally:
            os.close(fd)
    except Exception:
        pass


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
        fd = os.open(str(json_path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            os.write(fd, json.dumps(existing).encode("utf-8"))
        finally:
            os.close(fd)
    except Exception:
        pass


def _ext_to_tags(file_path: str) -> list:
    """Derive context tags from file extension for re-ranking boost."""
    ext = os.path.splitext(file_path)[1].lower()
    _EXT_MAP = {
        ".py": ["python"], ".js": ["javascript"], ".ts": ["typescript"],
        ".tsx": ["typescript", "react"], ".jsx": ["javascript", "react"],
        ".rs": ["rust"], ".go": ["go"], ".rb": ["ruby"],
        ".java": ["java"], ".swift": ["swift"], ".sh": ["bash"],
        ".sql": ["sql"], ".md": ["markdown"], ".yml": ["yaml"],
        ".yaml": ["yaml"], ".json": ["json"], ".toml": ["toml"],
        ".css": ["css"], ".html": ["html"], ".c": ["c"],
        ".cpp": ["c++"], ".vue": ["vue", "javascript"],
        ".svelte": ["svelte", "javascript"], ".tf": ["terraform"],
        ".graphql": ["graphql"], ".prisma": ["prisma"],
        ".env": ["config"], ".ini": ["config"],
        ".kt": ["kotlin"], ".sc": ["scala"], ".ex": ["elixir"],
        ".php": ["php"], ".r": ["r"],
    }
    return _EXT_MAP.get(ext, [])


def _lookup_session_tasks(results: list) -> dict:
    """Look up session task descriptions for memory results.

    Returns a dict mapping session_id -> task description (truncated).
    Uses coord_sessions table; returns empty dict on failure.
    """
    session_ids = {r.get("session_id") for r in results if r.get("session_id")}
    if not session_ids:
        return {}
    try:
        from omega.coordination import get_manager
        mgr = get_manager()
        placeholders = ",".join("?" for _ in session_ids)
        cursor = mgr._conn.execute(
            f"SELECT session_id, task FROM coord_sessions WHERE session_id IN ({placeholders})",
            list(session_ids),
        )
        return {row[0]: (row[1] or "")[:40] for row in cursor.fetchall() if row[1]}
    except Exception:
        return {}


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


def _surface_for_edit(file_path: str, session_id: str, project: str, count_surfacing: bool = True):
    """Surface memories related to a file being edited, with attribution."""
    try:
        from omega.bridge import query_structured
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
        )
        # Boost by capture confidence, then filter low-relevance noise
        results = _apply_confidence_boost(results)
        results = [r for r in results if r.get("relevance", 0.0) >= 0.20]
        if not results:
            return

        # Look up session task descriptions for source attribution
        session_tasks = _lookup_session_tasks(results)

        print(f"\n[MEMORY] Relevant context for {filename}:")
        for r in results:
            score = r.get("relevance", 0.0)
            etype = r.get("event_type", "memory")
            preview = r.get("content", "")[:120].replace('\n', ' ')
            nid = r.get("id", "")[:8]
            created = r.get("created_at", "")
            age = _relative_time_from_iso(created) if created else ""
            # Build attribution: "(2d ago, from 'implementing auth')" or "(2d ago)"
            mem_session = r.get("session_id", "")
            task_desc = session_tasks.get(mem_session, "")
            if age and task_desc:
                attr = f" ({age}, from \"{task_desc}\")"
            elif age:
                attr = f" ({age})"
            elif task_desc:
                attr = f" (from \"{task_desc}\")"
            else:
                attr = ""
            print(f"  [{score:.0%}] {etype}{attr}: {preview} (id:{nid})")

        # First-recall milestone
        try:
            if _check_milestone("first-recall"):
                print("[OMEGA] First memory recalled! Past context is informing this edit.")
        except Exception:
            pass

        # Traverse: surface linked memories from the top result
        try:
            if results and results[0].get("id"):
                from omega.bridge import _get_store
                store = _get_store()
                shown_ids = {r.get("id") for r in results}
                chain = store.get_related_chain(results[0]["id"], max_hops=1, min_weight=0.4)
                linked_count = 0
                for node in chain:
                    nid = node.get("node_id") or node.get("id", "")
                    if nid in shown_ids or not nid:
                        continue
                    etype = node.get("event_type", "memory")
                    preview = node.get("content", "")[:120].replace('\n', ' ')
                    print(f"  [linked] {etype}: {preview}")
                    linked_count += 1
                    if linked_count >= 2:
                        break
        except Exception:
            pass

        # Phrase search: exact-match error patterns for this file
        try:
            from omega.bridge import _get_store
            store = _get_store()
            filename = os.path.basename(file_path)
            exact_hits = store.phrase_search(filename, limit=2, event_type="error_pattern")
            shown_ids = {r.get("id") for r in results}
            for hit in exact_hits:
                hid = hit.get("node_id") or hit.get("id", "")
                if hid in shown_ids:
                    continue
                preview = hit.get("content", "")[:120].replace('\n', ' ')
                print(f"  [exact] error: {preview}")
        except Exception:
            pass

        # Track surfaced memory IDs for auto-feedback on session stop
        if count_surfacing and session_id:
            memory_ids = [r.get("id") for r in results if r.get("id")]
            _track_surfaced_ids(session_id, file_path, memory_ids)

        # Track surfacing count for session activity summary (edits only, not reads)
        if count_surfacing and session_id:
            try:
                marker = Path.home() / ".omega" / f"session-{session_id}.surfaced"
                with open(marker, "a") as f:
                    f.write("x")
            except Exception:
                pass
    except ImportError:
        pass
    except Exception as e:
        _log_hook_error("surface_for_edit", e)


def _surface_lessons(file_path: str, session_id: str, project: str):
    """Surface verified cross-session lessons relevant to a file."""
    try:
        from omega.bridge import get_cross_session_lessons
        filename = os.path.basename(file_path)
        lessons = get_cross_session_lessons(
            task=f"editing {filename}",
            project_path=project,
            exclude_session=session_id,
            limit=2,
        )
        verified = [l for l in lessons if l.get("verified")]
        if verified:
            print(f"\n[LESSON] Verified wisdom for {filename}:")
            for l in verified:
                content = l.get("content", "")[:150]
                print(f"  - {content}")
    except ImportError:
        pass
    except Exception as e:
        _log_hook_error("surface_lessons", e)



# Session-level error dedup cache
_error_hashes: set = set()
_error_count: int = 0
_MAX_ERRORS_PER_SESSION = 5


def _extract_error_summary(raw_output: str) -> str:
    """Extract a clean error summary from raw tool output.

    For tracebacks: grab the last non-frame line (the actual error).
    For JSON blobs: skip JSON structure, find the error marker line.
    Returns a clean summary string, or the raw content truncated.
    """
    lines = raw_output.strip().split("\n")

    # Traceback: find the last exception line (after "File ..." frames)
    if "Traceback (most recent call last)" in raw_output:
        for line in reversed(lines):
            stripped = line.strip()
            if stripped and not stripped.startswith("File ") and not stripped.startswith("^"):
                return stripped[:300]

    # JSON blob: skip lines starting with {, [, }, ], or pure whitespace/quotes
    non_json_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith(("{", "[", "}", "]", '"')):
            non_json_lines.append(stripped)
    if non_json_lines:
        # Find the first line with an error marker
        error_markers_local = [
            "Error:", "ERROR:", "error:", "FAILED", "Failed",
            "SyntaxError:", "TypeError:", "NameError:", "ImportError:",
            "ModuleNotFoundError:", "AttributeError:", "ValueError:",
            "KeyError:", "IndexError:", "FileNotFoundError:",
            "fatal:", "FATAL:", "panic:",
            "command not found", "No such file or directory",
            "Permission denied", "Connection refused",
        ]
        for line in non_json_lines:
            if any(m in line for m in error_markers_local):
                return line[:300]
        return non_json_lines[0][:300]

    return raw_output[:300]


def _capture_error(tool_output: str, session_id: str, project: str):
    """Auto-capture error patterns from Bash tool failures.

    Session-level dedup: skip if same error pattern already captured.
    Cap at _MAX_ERRORS_PER_SESSION to prevent test-run floods.
    """
    global _error_count

    if not tool_output:
        return
    if not isinstance(tool_output, str):
        tool_output = str(tool_output)

    # Cap errors per session
    if _error_count >= _MAX_ERRORS_PER_SESSION:
        return

    error_markers = [
        "Error:", "ERROR:", "error:", "FAILED", "Failed",
        "Traceback (most recent call last)",
        "SyntaxError:", "TypeError:", "NameError:", "ImportError:",
        "ModuleNotFoundError:", "AttributeError:", "ValueError:",
        "KeyError:", "IndexError:", "FileNotFoundError:",
        "fatal:", "FATAL:", "panic:",
        "command not found", "No such file or directory",
        "Permission denied", "Connection refused",
    ]

    has_error = any(marker in tool_output for marker in error_markers)
    if not has_error:
        return

    # Extract clean error summary instead of storing raw blob
    error_summary = _extract_error_summary(tool_output)

    # Session-level dedup: hash the first 100 chars (normalized)
    error_hash = re.sub(r'\s+', ' ', error_summary[:100].lower()).strip()
    if error_hash in _error_hashes:
        return
    _error_hashes.add(error_hash)
    _error_count += 1

    # --- "You've seen this before" — proactive error recall ---
    try:
        from omega.bridge import query_structured
        past_errors = query_structured(
            query_text=error_summary[:200],
            limit=2,
            project=project,
            event_type="error_pattern",
        )
        past_lessons = query_structured(
            query_text=error_summary[:200],
            limit=2,
            event_type="lesson_learned",
        )
        past_matches = []
        for m in (past_errors or []) + (past_lessons or []):
            if m.get("relevance", 0) >= 0.35 and m.get("session_id") != session_id:
                past_matches.append(m)
        if past_matches:
            print("\n[RECALL] You've seen this before:")
            for m in past_matches[:2]:
                etype = m.get("event_type", "memory")
                content = m.get("content", "")[:150].replace("\n", " ").strip()
                print(f"  [{etype}] {content}")
    except ImportError:
        pass
    except Exception as e:
        _log_hook_error("error_recall", e)

    try:
        from omega.bridge import auto_capture
        result = auto_capture(
            content=f"Error: {error_summary}",
            event_type="error_pattern",
            metadata={
                "source": "auto_capture_hook",
                "project": project,
                "capture_confidence": "medium",
            },
            session_id=session_id,
            project=project,
        )
        if result and ("Memory Captured" in result or "Memory Evolved" in result):
            first_line = error_summary.split('\n')[0][:80]
            if "Evolved" in result:
                evo_match = re.search(r"Evolution #(\d+)", result)
                evo_num = evo_match.group(1) if evo_match else "?"
                print(f"[OMEGA] Memory evolved: error pattern updated (evolution #{evo_num}) — {first_line}")
            else:
                print(f"[OMEGA] Captured: error — {first_line}")
    except ImportError:
        pass
    except Exception as e:
        _log_hook_error("capture_error", e)


def _track_git_commit(tool_input: str, tool_output: str, session_id: str, project: str):
    """Detect git commit in Bash output and log to coordination."""
    if not tool_output:
        return
    if not isinstance(tool_output, str):
        tool_output = str(tool_output)

    # Check if the command was a git commit
    try:
        input_data = json.loads(tool_input)
    except (json.JSONDecodeError, TypeError):
        return

    command = input_data.get("command", "")
    if "git commit" not in command:
        return

    # Parse commit hash from output — git outputs lines like:
    #   [main abc1234] Commit message here
    match = re.search(r'\[[\w/.-]+\s+([0-9a-f]{7,12})\]', tool_output)
    if not match:
        return

    commit_hash = match.group(1)

    # Extract commit message from the same line
    msg_match = re.search(r'\[[\w/.-]+\s+[0-9a-f]{7,12}\]\s+(.+)', tool_output)
    message = msg_match.group(1).strip() if msg_match else ""

    # Get current branch
    import subprocess
    try:
        branch_result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True, text=True, timeout=5, cwd=project,
        )
        branch = branch_result.stdout.strip() if branch_result.returncode == 0 else None
    except Exception:
        branch = None

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
    except ImportError:
        pass
    except Exception as e:
        _log_hook_error("track_git_commit", e)


def main():
    tool_name = os.environ.get("TOOL_NAME", "")
    tool_input = os.environ.get("TOOL_INPUT", "{}")
    tool_output = os.environ.get("TOOL_OUTPUT", "")
    session_id = os.environ.get("SESSION_ID", "")
    project = os.environ.get("PROJECT_DIR", os.getcwd())

    # Surface memories on file edits
    if tool_name in ("Edit", "Write", "NotebookEdit"):
        try:
            input_data = json.loads(tool_input)
        except (json.JSONDecodeError, TypeError):
            return

        file_path = input_data.get("file_path", input_data.get("notebook_path", ""))
        if file_path:
            _surface_for_edit(file_path, session_id, project)
            _surface_lessons(file_path, session_id, project)

    # Surface memories on file reads (lightweight — no lessons)
    if tool_name == "Read":
        try:
            input_data = json.loads(tool_input)
        except (json.JSONDecodeError, TypeError):
            input_data = {}

        file_path = input_data.get("file_path", "")
        if file_path:
            _surface_for_edit(file_path, session_id, project, count_surfacing=False)

    # Auto-capture errors from Bash failures + track git commits
    if tool_name == "Bash" and tool_output:
        _capture_error(tool_output, session_id, project)
        _track_git_commit(tool_input, tool_output, session_id, project)



if __name__ == "__main__":
    _t0 = time.monotonic()
    main()
    _log_timing("surface_memories", (time.monotonic() - _t0) * 1000)
