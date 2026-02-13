#!/usr/bin/env python3
"""OMEGA fast hook client — routes to daemon via UDS, falls back to direct.

This is a thin client that connects to the hook server running inside the
MCP process. It avoids importing any OMEGA modules on the fast path, keeping
startup to ~50ms (Python interpreter only).

If the daemon socket is unavailable (MCP not started yet, or crashed),
falls back to running the original hook script directly.
"""
import json
import os
import socket
import sys
import time

SOCK_PATH = os.path.expanduser("~/.omega/hook.sock")

# Map hook names to their original script modules for fallback
_FALLBACK_SCRIPTS = {
    "session_start": "session_start",
    "session_stop": "session_stop",
    "surface_memories": "surface_memories",
    "auto_capture": "auto_capture",
    "coord_session_start": "coord_session_start",
    "coord_session_stop": "coord_session_stop",
    "coord_heartbeat": "coord_heartbeat",
    "auto_claim_file": "auto_claim_file",
    "pre_file_guard": "pre_file_guard",
    "pre_task_guard": "pre_task_guard",
    "pre_push_guard": "pre_push_guard",
}

# Hooks that require longer timeouts (e.g., git network operations)
_SLOW_HOOKS = {"pre_push_guard"}


def delegate(hook_names, payload, timeout=5.0):
    """Connect to daemon, send request, return parsed response.

    Accepts a single hook name (str) or multiple (list) for batching.
    Batch requests use {"hooks": [...]} and return {"results": [...]}.
    """
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(timeout)
    try:
        s.connect(SOCK_PATH)
        if isinstance(hook_names, list):
            request = json.dumps({"hooks": hook_names, **payload}).encode("utf-8")
        else:
            request = json.dumps({"hook": hook_names, **payload}).encode("utf-8")
        s.sendall(request)
        s.shutdown(socket.SHUT_WR)

        response = b""
        while True:
            chunk = s.recv(8192)
            if not chunk:
                break
            response += chunk
        return json.loads(response.decode("utf-8"))
    finally:
        s.close()


def _fallback(hook_name, payload):
    """Run the original hook script directly (cold path).

    Sets env vars from *payload* before calling mod.main() so that
    individual hook scripts (which read os.environ) see the values
    that Claude Code passed via stdin JSON.
    """
    hooks_dir = os.path.dirname(os.path.abspath(__file__))
    script_name = _FALLBACK_SCRIPTS.get(hook_name)
    if not script_name:
        return

    script_path = os.path.join(hooks_dir, f"{script_name}.py")
    if not os.path.exists(script_path):
        return

    # Bridge: set env vars from payload so hook scripts can read them.
    # Claude Code stdin JSON uses different field names than env vars:
    #   stdin: session_id, tool_name, tool_input (dict), tool_response, cwd
    #   env:   SESSION_ID, TOOL_NAME, TOOL_INPUT (str),  TOOL_OUTPUT,  PROJECT_DIR
    _ENV_MAP = {
        "session_id": "SESSION_ID",
        "tool_name": "TOOL_NAME",
        "tool_input": "TOOL_INPUT",
        "tool_response": "TOOL_OUTPUT",  # Claude Code calls it tool_response
        "tool_output": "TOOL_OUTPUT",    # legacy/internal name
        "cwd": "PROJECT_DIR",
        "project": "PROJECT_DIR",        # internal name used by some hooks
    }
    for payload_key, env_key in _ENV_MAP.items():
        val = payload.get(payload_key)
        if val:
            # tool_input/tool_response may be dicts from JSON parse — serialize
            if isinstance(val, (dict, list)):
                os.environ[env_key] = json.dumps(val)
            else:
                os.environ[env_key] = str(val)

    # Add hooks dir to path so the script can be imported
    if hooks_dir not in sys.path:
        sys.path.insert(0, hooks_dir)

    import importlib
    try:
        mod = importlib.import_module(script_name)
        if hasattr(mod, "main"):
            mod.main()
    except Exception as e:
        print(f"OMEGA hook fallback error ({hook_name}): {e}", file=sys.stderr)


def _log_timing(hook_name, elapsed_ms, mode):
    """Log hook timing to ~/.omega/hooks.log."""
    try:
        from datetime import datetime
        from pathlib import Path
        log_path = Path.home() / ".omega" / "hooks.log"
        log_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        timestamp = datetime.now().isoformat(timespec="seconds")
        data = f"[{timestamp}] fast_hook/{hook_name}: OK ({elapsed_ms:.0f}ms, {mode})\n"
        fd = os.open(str(log_path), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            os.write(fd, data.encode("utf-8"))
        finally:
            os.close(fd)
    except Exception:
        pass


def _parse_payload():
    """Build payload from env vars + stdin JSON."""
    payload = {
        "tool_name": os.environ.get("TOOL_NAME", ""),
        "tool_input": os.environ.get("TOOL_INPUT", "{}"),
        "tool_output": os.environ.get("TOOL_OUTPUT", ""),
        "session_id": os.environ.get("SESSION_ID", ""),
        "project": os.environ.get("PROJECT_DIR", os.getcwd()),
    }

    # Claude Code sends hook data as JSON on stdin for ALL hook types.
    if not sys.stdin.isatty():
        try:
            raw = sys.stdin.read()
            if raw.strip():
                try:
                    stdin_data = json.loads(raw)
                    if isinstance(stdin_data, dict):
                        if "tool_response" in stdin_data and "tool_output" not in stdin_data:
                            stdin_data["tool_output"] = stdin_data["tool_response"]
                        if "cwd" in stdin_data and "project" not in stdin_data:
                            stdin_data["project"] = stdin_data["cwd"]
                        if isinstance(stdin_data.get("tool_input"), (dict, list)):
                            stdin_data["tool_input"] = json.dumps(stdin_data["tool_input"])
                        if isinstance(stdin_data.get("tool_output"), (dict, list)):
                            stdin_data["tool_output"] = json.dumps(stdin_data["tool_output"])
                        for key, val in stdin_data.items():
                            if val or not payload.get(key):
                                payload[key] = val
                except json.JSONDecodeError:
                    payload["stdin"] = raw
        except Exception:
            pass

    return payload


def main():
    if len(sys.argv) < 2:
        print("Usage: fast_hook.py <hook_name[+hook_name...]>", file=sys.stderr)
        sys.exit(1)

    t0 = time.monotonic()
    hook_names = sys.argv[1].split("+")
    payload = _parse_payload()
    is_batch = len(hook_names) > 1

    # Use longer timeout for hooks with network operations (e.g., git fetch)
    timeout = 5.0
    if _SLOW_HOOKS.intersection(hook_names):
        timeout = 20.0

    try:
        result = delegate(hook_names if is_batch else hook_names[0], payload, timeout=timeout)
        elapsed_ms = (time.monotonic() - t0) * 1000

        if is_batch:
            # Batch response: {"results": [{output, error, exit_code}, ...]}
            outputs = []
            exit_code = 0
            for r in result.get("results", []):
                if r.get("output"):
                    outputs.append(r["output"])
                if r.get("exit_code") and not exit_code:
                    exit_code = r["exit_code"]
            if outputs:
                print("\n".join(outputs))
            _log_timing("+".join(hook_names), elapsed_ms, "daemon")
            if exit_code:
                sys.exit(exit_code)
        else:
            if result.get("output"):
                print(result["output"])
            _log_timing(hook_names[0], elapsed_ms, "daemon")
            exit_code = result.get("exit_code")
            if exit_code:
                sys.exit(exit_code)

    except (ConnectionRefusedError, FileNotFoundError, socket.timeout, OSError):
        # Daemon not running — fall back to direct execution (run each sequentially).
        # NOTE: If a blocking hook calls sys.exit(2), SystemExit propagates through
        # and kills the process — correctly preventing subsequent hooks from running.
        for name in hook_names:
            _fallback(name, payload)
        elapsed_ms = (time.monotonic() - t0) * 1000
        _log_timing("+".join(hook_names), elapsed_ms, "fallback")


if __name__ == "__main__":
    main()
