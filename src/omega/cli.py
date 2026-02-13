"""OMEGA CLI — Memory commands, setup, status, migration, and server management."""

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

OMEGA_DIR = Path.home() / ".omega"
OMEGA_CACHE = Path.home() / ".cache" / "omega"
MAGMA_DIR = Path.home() / ".magma"
MAGMA_GRAPHS = Path.home() / ".claude" / "magma"
BGE_MODEL_DIR = OMEGA_CACHE / "models" / "bge-small-en-v1.5-onnx"
MINILM_MODEL_DIR = OMEGA_CACHE / "models" / "all-MiniLM-L6-v2-onnx"
# Primary model dir — bge-small-en-v1.5, falls back to all-MiniLM-L6-v2
ONNX_MODEL_DIR = BGE_MODEL_DIR


CLAUDE_MD_PATH = Path.home() / ".claude" / "CLAUDE.md"
SETTINGS_JSON_PATH = Path.home() / ".claude" / "settings.json"
DATA_DIR = Path(__file__).parent / "data"

OMEGA_BEGIN = "<!-- OMEGA:BEGIN"
OMEGA_END = "<!-- OMEGA:END -->"


def _resolve_python_path() -> str:
    """Resolve the best Python interpreter path for hooks.

    Priority:
    1. sys.executable if it exists and is not inside a temporary venv
    2. 'python3' from PATH (via shutil.which)
    3. Hardcoded /opt/homebrew/bin/python3 as last resort (macOS)
    """
    exe = sys.executable
    if exe and Path(exe).exists() and "venv" not in exe:
        return exe
    which_py = shutil.which("python3")
    if which_py:
        return which_py
    # Last resort for macOS Homebrew
    fallback = "/opt/homebrew/bin/python3"
    if Path(fallback).exists():
        return fallback
    return exe or "python3"


def _inject_claude_md():
    """Inject or update the OMEGA block in ~/.claude/CLAUDE.md (idempotent)."""
    fragment = (DATA_DIR / "claude-md-fragment.md").read_text()

    if CLAUDE_MD_PATH.exists():
        content = CLAUDE_MD_PATH.read_text()
    else:
        CLAUDE_MD_PATH.parent.mkdir(parents=True, exist_ok=True)
        content = ""

    if OMEGA_BEGIN in content:
        # Replace existing block (upgrade path)
        pattern = re.compile(
            r"<!-- OMEGA:BEGIN[^\n]*-->.*?<!-- OMEGA:END -->",
            re.DOTALL,
        )
        new_content = pattern.sub(fragment.rstrip(), content)
        if new_content == content:
            print("  CLAUDE.md: OMEGA block already up to date")
            return
        CLAUDE_MD_PATH.write_text(new_content)
        print("  CLAUDE.md: OMEGA block updated")
    else:
        # Append — check if there's a plain "## Memory (OMEGA)" section to replace
        plain_pattern = re.compile(r"## Memory \(OMEGA\)\n(?:.*\n)*?(?=\n## |\Z)", re.MULTILINE)
        if plain_pattern.search(content):
            new_content = plain_pattern.sub(fragment.rstrip() + "\n", content)
            CLAUDE_MD_PATH.write_text(new_content)
            print("  CLAUDE.md: replaced plain Memory section with managed block")
        else:
            separator = "\n" if content and not content.endswith("\n") else ""
            CLAUDE_MD_PATH.write_text(content + separator + fragment)
            print("  CLAUDE.md: OMEGA block appended")


def _has_commercial_modules() -> bool:
    """Check if commercial/coordination modules are available."""
    try:
        import omega.coordination  # noqa: F401

        return True
    except ImportError:
        pass
    try:
        from omega.plugins import discover_plugins

        for plugin in discover_plugins():
            if plugin.HOOKS_JSON:
                return True
    except Exception:
        pass
    return False


def _inject_settings_hooks(hooks_src: Path):
    """Inject OMEGA hook entries into ~/.claude/settings.json (idempotent).

    Uses hooks-core.json for core-only installs, or hooks.json (full) when
    commercial modules are available. Supports both old format (single dict
    per event) and new format (list of dicts per event) in hooks.json manifest.
    """
    if _has_commercial_modules():
        hooks_file = "hooks.json"
    else:
        hooks_file = "hooks-core.json"
    manifest = json.loads((DATA_DIR / hooks_file).read_text())

    # Determine the python path: prefer the running interpreter
    python_path = _resolve_python_path()

    if SETTINGS_JSON_PATH.exists():
        try:
            settings = json.loads(SETTINGS_JSON_PATH.read_text())
        except json.JSONDecodeError:
            print("  WARNING: settings.json is malformed, skipping hook injection")
            return
    else:
        SETTINGS_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
        settings = {}

    if "hooks" not in settings:
        settings["hooks"] = {}

    configured = 0
    skipped = 0

    for event, hook_defs in manifest.items():
        # Normalize: old format is a single dict, new format is a list of dicts
        if isinstance(hook_defs, dict):
            hook_defs = [hook_defs]

        for hook_def in hook_defs:
            script = hook_def["script"]
            command = f"{python_path} {hooks_src / script}"

            # Build a unique identifier for this hook (handles "fast_hook.py session_start" etc.)
            # Strip .py and use the full script string for matching
            script_key = script.replace(".py", "").replace(" ", "_")

            # Check if this OMEGA hook is already wired
            already_wired = False
            if event in settings["hooks"]:
                for entry in settings["hooks"][event]:
                    for h in entry.get("hooks", []):
                        cmd = h.get("command", "")
                        if "omega" in cmd and script_key in cmd.replace(".py", "").replace(" ", "_"):
                            already_wired = True
                            break
                    if already_wired:
                        break

            if already_wired:
                skipped += 1
                continue

            # Build the hook entry
            entry = {
                "hooks": [
                    {
                        "command": command,
                        "timeout": hook_def["timeout"],
                        "type": "command",
                    }
                ],
                "matcher": hook_def.get("matcher", ""),
            }

            if event not in settings["hooks"]:
                settings["hooks"][event] = []
            settings["hooks"][event].append(entry)
            configured += 1

    SETTINGS_JSON_PATH.write_text(json.dumps(settings, indent=2) + "\n")

    if configured > 0:
        print(f"  settings.json: {configured} hook(s) configured")
    if skipped > 0:
        print(f"  settings.json: {skipped} hook(s) already configured")
    if configured == 0 and skipped == 0:
        print("  settings.json: hooks configured")


def _download_file(url: str, target: Path) -> None:
    """Download a file with a progress bar showing bytes and percentage."""
    import urllib.request

    req = urllib.request.Request(url, headers={"User-Agent": "omega-memory/1.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        total = int(resp.headers.get("Content-Length", 0))
        downloaded = 0
        chunk_size = 64 * 1024  # 64 KB chunks

        # Write to a temp file, rename on success (no partial files left behind)
        tmp = target.with_suffix(target.suffix + ".tmp")
        try:
            with open(tmp, "wb") as f:
                while True:
                    chunk = resp.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total > 0:
                        pct = downloaded * 100 // total
                        mb_done = downloaded / (1024 * 1024)
                        mb_total = total / (1024 * 1024)
                        print(f"\r    {target.name}: {mb_done:.1f}/{mb_total:.1f} MB ({pct}%)", end="", flush=True)
                    else:
                        mb_done = downloaded / (1024 * 1024)
                        print(f"\r    {target.name}: {mb_done:.1f} MB", end="", flush=True)
            tmp.rename(target)
            print()  # newline after progress
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise


def _download_bge_model(target_dir: Path, errors_ref: list) -> bool:
    """Download bge-small-en-v1.5 ONNX model from HuggingFace. Returns True on success."""
    target_dir.mkdir(parents=True, exist_ok=True)
    required = ["model.onnx", "tokenizer.json", "config.json"]
    if all((target_dir / f).exists() for f in required):
        print(f"  bge-small-en-v1.5 model already present at {target_dir}")
        return True

    print("  Downloading bge-small-en-v1.5 ONNX model (~130MB)...")
    try:
        hf_repo = "https://huggingface.co/BAAI/bge-small-en-v1.5/resolve/main"
        # model.onnx lives in onnx/ subdir, tokenizer files at repo root
        files = {
            "model.onnx": f"{hf_repo}/onnx/model.onnx",
            "tokenizer.json": f"{hf_repo}/tokenizer.json",
            "config.json": f"{hf_repo}/config.json",
            "tokenizer_config.json": f"{hf_repo}/tokenizer_config.json",
        }
        for fname, url in files.items():
            target = target_dir / fname
            if not target.exists():
                _download_file(url, target)
    except Exception as e:
        errors_ref.append(e)
        print(f"  ERROR: bge model download failed: {e}")
        print(f"  Manually place model files in {target_dir}")
        return False

    if not (target_dir / "model.onnx").exists():
        errors_ref.append("model.onnx not present after download")
        print("  ERROR: model.onnx still not present after download attempt")
        return False
    print(f"  bge-small-en-v1.5 model downloaded to {target_dir}")
    return True


# ---------------------------------------------------------------------------
# CLI Memory Commands — direct terminal access to OMEGA
# ---------------------------------------------------------------------------


def _format_age(created_at) -> str:
    """Format a datetime as relative age string (e.g. '2d ago', '1w ago')."""
    if not created_at:
        return ""
    now = datetime.now(timezone.utc)
    if created_at.tzinfo is None:
        # Naive datetime — assume UTC
        created_at = created_at.replace(tzinfo=timezone.utc)
    delta = now - created_at
    seconds = int(delta.total_seconds())
    if seconds < 60:
        return "just now"
    if seconds < 3600:
        return f"{seconds // 60}m ago"
    if seconds < 86400:
        return f"{seconds // 3600}h ago"
    days = seconds // 86400
    if days < 7:
        return f"{days}d ago"
    if days < 30:
        return f"{days // 7}w ago"
    return f"{days // 30}mo ago"


def cmd_query(args):
    """Search memories by semantic similarity or exact phrase."""
    query_text = " ".join(args.query_text)
    if not query_text.strip():
        print("Usage: omega query <search text>", file=sys.stderr)
        sys.exit(1)

    limit = getattr(args, "limit", 10)
    use_json = getattr(args, "json", False)
    exact = getattr(args, "exact", False)

    start = time.monotonic()

    if exact:
        # For --json, use the store directly
        if use_json:
            from omega.bridge import _get_store

            db = _get_store()
            results = db.phrase_search(phrase=query_text, limit=limit)
            elapsed = time.monotonic() - start
            out = []
            for node in results:
                out.append(
                    {
                        "id": node.id,
                        "content": node.content,
                        "event_type": (node.metadata or {}).get("event_type", "memory"),
                        "created_at": node.created_at.isoformat() if node.created_at else "",
                        "tags": (node.metadata or {}).get("tags", []),
                    }
                )
            print(json.dumps({"results": out, "count": len(out), "elapsed_s": round(elapsed, 3)}, indent=2))
        else:
            from omega.bridge import _get_store

            db = _get_store()
            results = db.phrase_search(phrase=query_text, limit=limit)
            elapsed = time.monotonic() - start
            if results:
                from omega.cli_ui import print_table

                rows = []
                for node in results:
                    etype = (node.metadata or {}).get("event_type", "memory")
                    preview = node.content[:120].replace("\n", " ")
                    age = _format_age(node.created_at)
                    mid = node.id[:12] if node.id else ""
                    rows.append(("--", etype, preview, age, mid))
                print_table(
                    None, ["Score", "Type", "Preview", "Age", "ID"], rows, styles=["dim", "bold", None, "dim", "dim"]
                )
                print(f"\n{len(results)} result(s) ({elapsed:.2f}s)")
            else:
                print(f'No results for "{query_text}" ({elapsed:.2f}s)')
    else:
        from omega.bridge import query_structured

        results = query_structured(query_text, limit=limit)
        elapsed = time.monotonic() - start

        if use_json:
            print(json.dumps({"results": results, "count": len(results), "elapsed_s": round(elapsed, 3)}, indent=2))
        else:
            if results:
                from omega.cli_ui import print_table

                rows = []
                for r in results:
                    relevance = f"{int(r.get('relevance', 0) * 100)}%"
                    etype = r.get("event_type", "memory")
                    preview = r.get("content", "")[:120].replace("\n", " ")
                    age = ""
                    if r.get("created_at"):
                        try:
                            dt = datetime.fromisoformat(r["created_at"])
                            age = _format_age(dt)
                        except (ValueError, TypeError):
                            pass
                    mid = r.get("id", "")[:12]
                    rows.append((relevance, etype, preview, age, mid))
                print_table(
                    None, ["Score", "Type", "Preview", "Age", "ID"], rows, styles=["cyan", "bold", None, "dim", "dim"]
                )
                print(f"\n{len(results)} result(s) ({elapsed:.2f}s)")
            else:
                print(f'No results for "{query_text}" ({elapsed:.2f}s)')


_CLI_TYPE_MAP = {
    "memory": "memory",
    "lesson": "lesson_learned",
    "decision": "decision",
    "error": "error_pattern",
    "task": "task_completion",
    "preference": "user_preference",
}


def cmd_store(args):
    """Store a memory with a specified type."""
    content = " ".join(args.content)
    if not content.strip():
        print("Usage: omega store <text> [-t TYPE]", file=sys.stderr)
        sys.exit(1)

    cli_type = getattr(args, "type", "memory")
    event_type = _CLI_TYPE_MAP.get(cli_type, cli_type)

    from omega.bridge import store

    store(content=content, event_type=event_type)
    print(f"Stored [{cli_type}]: {content[:80]}")


def cmd_remember(args):
    """Store a permanent user preference."""
    text = " ".join(args.text)
    if not text.strip():
        print("Usage: omega remember <text>", file=sys.stderr)
        sys.exit(1)

    from omega.bridge import remember

    remember(text=text)
    print(f"Remembered: {text[:120]}")


def cmd_timeline(args):
    """Show memory timeline grouped by day."""
    days = getattr(args, "days", 7)
    use_json = getattr(args, "json", False)

    if use_json:
        from omega.bridge import _get_store

        db = _get_store()
        data = db.get_timeline(days=days, limit_per_day=20)
        out = {}
        for day, memories in (data or {}).items():
            out[day] = []
            for m in memories:
                out[day].append(
                    {
                        "id": m.id,
                        "content": m.content[:200],
                        "event_type": (m.metadata or {}).get("event_type", "memory"),
                        "created_at": m.created_at.isoformat() if m.created_at else "",
                    }
                )
        print(json.dumps(out, indent=2))
    else:
        from omega.bridge import _get_store
        from omega.cli_ui import print_header, print_table

        db = _get_store()
        data = db.get_timeline(days=days, limit_per_day=20)
        if not data:
            print(f"No memories in the last {days} days.")
            return

        total = sum(len(v) for v in data.values())
        print_header(f"Memory Timeline ({total} memories, last {days} days)")

        for day in sorted(data.keys(), reverse=True):
            memories = data[day]
            rows = []
            for m in memories:
                etype = (m.metadata or {}).get("event_type", "memory")
                preview = m.content[:100].replace("\n", " ")
                time_str = m.created_at.strftime("%H:%M") if m.created_at else ""
                mid = m.id[:12] if m.id else ""
                rows.append((time_str, etype, preview, mid))
            print_table(
                f"{day} ({len(memories)})",
                ["Time", "Type", "Preview", "ID"],
                rows,
                styles=["dim", "bold", None, "dim"],
            )


# ---------------------------------------------------------------------------
# Setup & Doctor
# ---------------------------------------------------------------------------


def _setup_claude_code(errors_ref: list, hooks_src: Path):
    """Claude Code-specific setup: MCP registration, hooks, CLAUDE.md."""
    # Register MCP server with Claude Code
    print("  Registering MCP server with Claude Code...")
    python_path = _resolve_python_path()
    try:
        result = subprocess.run(
            ["claude", "mcp", "add", "omega-memory", "--", python_path, "-m", "omega.server.mcp_server"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            print("  MCP server registered successfully")
        else:
            errors_ref.append(1)
            print(f"  ERROR: MCP registration returned code {result.returncode}")
            if result.stderr:
                print(f"  {result.stderr.strip()}")
            print(f"  Register manually: claude mcp add omega-memory -- {python_path} -m omega.server.mcp_server")
    except FileNotFoundError:
        errors_ref.append(1)
        print("  ERROR: 'claude' command not found in PATH.")
        print("  Install Claude Code: https://docs.anthropic.com/en/docs/claude-code")
        print(f"  Or register manually: claude mcp add omega-memory -- {python_path} -m omega.server.mcp_server")
    except Exception as e:
        errors_ref.append(1)
        print(f"  ERROR: MCP registration failed: {e}")
        print(f"  Register manually: claude mcp add omega-memory -- {python_path} -m omega.server.mcp_server")

    # Install hooks
    hooks_dst = Path.home() / ".claude" / "scripts"
    hooks_dst.mkdir(parents=True, exist_ok=True)

    hook_files = ["session_start.py", "session_stop.py", "surface_memories.py", "auto_capture.py"]
    for hook in hook_files:
        src = hooks_src / hook
        dst = hooks_dst / f"omega-{hook}"
        if src.exists():
            shutil.copy2(src, dst)
            dst.chmod(0o755)
            print(f"  Installed hook: {dst.name}")
        else:
            print(f"  WARNING: Hook source not found: {src}")

    # Wire hooks into settings.json
    try:
        _inject_settings_hooks(hooks_src)
    except Exception as e:
        errors_ref.append(1)
        print(f"  ERROR: Failed to configure settings.json hooks: {e}")

    # Inject OMEGA block into CLAUDE.md
    try:
        _inject_claude_md()
    except Exception as e:
        print(f"  WARNING: Failed to update CLAUDE.md: {e}")


def cmd_setup(args):
    """Set up OMEGA: create dirs, download model, initialize DB. Optionally configure a client."""
    # ── Python version check ──────────────────────────────────────────
    if sys.version_info < (3, 11):
        print(f"ERROR: OMEGA requires Python 3.11 or higher (you have {sys.version_info.major}.{sys.version_info.minor}).")
        print("Install Python 3.11+: https://www.python.org/downloads/")
        sys.exit(1)

    client = getattr(args, "client", None)
    errors = []
    download_model = getattr(args, "download_model", False)

    # ── Auto-detect Claude Code if --client not specified ─────────────
    if client is None and shutil.which("claude"):
        client = "claude-code"
        print("Setting up OMEGA (Claude Code detected)...")
    elif client is None:
        print("Setting up OMEGA...")
        print("  NOTE: Claude Code CLI not found in PATH.")
        print("  Skipping MCP registration and hooks. To add them later:")
        print("    omega setup --client claude-code")
        print()
    else:
        print("Setting up OMEGA...")

    # Track what we did for the summary
    steps_done = []
    steps_skipped = []
    files_modified = []

    # 1. Create directories with restricted permissions
    OMEGA_DIR.mkdir(parents=True, exist_ok=True, mode=0o700)
    (OMEGA_DIR / "graphs").mkdir(exist_ok=True, mode=0o700)
    print(f"  Created {OMEGA_DIR}")
    steps_done.append("Storage directory")

    # 2. Download ONNX model
    if download_model:
        _download_bge_model(BGE_MODEL_DIR, errors)
        steps_done.append("Embedding model (bge-small-en-v1.5)")
    else:
        bge_model = BGE_MODEL_DIR / "model.onnx"
        minilm_model = MINILM_MODEL_DIR / "model.onnx"
        if bge_model.exists():
            print(f"  ONNX model: bge-small-en-v1.5 at {BGE_MODEL_DIR}")
            steps_done.append("Embedding model (already present)")
        elif minilm_model.exists():
            print(f"  ONNX model: all-MiniLM-L6-v2 at {MINILM_MODEL_DIR}")
            print("  TIP: Run 'omega setup --download-model' to upgrade to bge-small-en-v1.5")
            steps_done.append("Embedding model (already present)")
        else:
            MINILM_MODEL_DIR.mkdir(parents=True, exist_ok=True)
            model_path = MINILM_MODEL_DIR / "model.onnx"
            print("  Downloading ONNX embedding model (all-MiniLM-L6-v2, ~90MB)...")
            script = Path(__file__).parent.parent.parent / "scripts" / "download_model.py"
            if script.exists():
                subprocess.run([sys.executable, str(script), str(MINILM_MODEL_DIR)], check=True)
            else:
                try:
                    hf_base = "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/onnx"
                    for fname in ["model.onnx", "tokenizer.json", "config.json", "tokenizer_config.json", "vocab.txt"]:
                        target = MINILM_MODEL_DIR / fname
                        if not target.exists():
                            _download_file(f"{hf_base}/{fname}", target)
                except Exception as e:
                    errors.append(e)
                    print(f"  ERROR: Model download failed: {e}")
                    print(f"  Manually place model files in {MINILM_MODEL_DIR}")
            if not model_path.exists():
                errors.append("model.onnx not present")
                print("  ERROR: model.onnx still not present after download attempt")
            else:
                print("  TIP: Run 'omega setup --download-model' to upgrade to bge-small-en-v1.5")
                steps_done.append("Embedding model (downloaded)")

    # 3. Check for existing MAGMA model and symlink
    gnosis_model = Path.home() / ".cache" / "gnosis" / "models" / "all-MiniLM-L6-v2-onnx"
    minilm_model_path = MINILM_MODEL_DIR / "model.onnx"
    if gnosis_model.exists() and not minilm_model_path.exists() and not (BGE_MODEL_DIR / "model.onnx").exists():
        print(f"  Found existing model at {gnosis_model}, creating symlink...")
        if MINILM_MODEL_DIR.exists():
            shutil.rmtree(MINILM_MODEL_DIR)
        MINILM_MODEL_DIR.symlink_to(gnosis_model)
        print("  Symlinked to existing model")

    # 4. Create default config
    config_path = OMEGA_DIR / "config.json"
    if not config_path.exists():
        config = {
            "storage_path": str(OMEGA_DIR),
            "model_dir": str(ONNX_MODEL_DIR),
            "version": "0.1.0",
            "entity_scoping": {"enabled": False},
        }
        config_path.write_text(json.dumps(config, indent=2))
        print(f"  Created config at {config_path}")
    steps_done.append("Config file")

    # 5. Client-specific setup
    hooks_src = Path(__file__).parent.parent.parent / "hooks"
    if client == "claude-code":
        _setup_claude_code(errors, hooks_src)
        steps_done.append("MCP server registration")
        steps_done.append("Hooks (settings.json)")
        steps_done.append("CLAUDE.md instructions")
        files_modified.extend([
            "~/.claude.json (MCP server entry)",
            "~/.claude/settings.json (hook entries)",
            "~/.claude/CLAUDE.md (OMEGA instruction block)",
        ])
    else:
        steps_skipped.append("MCP server registration (no Claude Code)")
        steps_skipped.append("Hooks (no Claude Code)")
        steps_skipped.append("CLAUDE.md instructions (no Claude Code)")
        python_path = _resolve_python_path()
        print("\n  MCP server ready. Add to your client:")
        print(f"    Command: {python_path} -m omega.server.mcp_server")
        print("    Transport: stdio")

    # ── Summary ───────────────────────────────────────────────────────
    print()
    if errors:
        print(f"OMEGA setup completed with {len(errors)} error(s).")
        for step in steps_done:
            print(f"  [OK] {step}")
        for err in errors:
            print(f"  [FAIL] {err}")
        for step in steps_skipped:
            print(f"  [SKIP] {step}")
        print("\nRun 'omega doctor' to diagnose issues.")
        sys.exit(1)
    else:
        print("OMEGA setup complete!")
        for step in steps_done:
            print(f"  [OK] {step}")
        for step in steps_skipped:
            print(f"  [SKIP] {step}")
        if files_modified:
            print("\n  Files modified outside ~/.omega/:")
            for f in files_modified:
                print(f"    {f}")
        print(f"\n  Storage: {OMEGA_DIR}")
        print("  Run 'omega doctor' to verify.")


def cmd_status(args):
    """Show OMEGA status: memory count, store size, model status."""
    from omega.cli_ui import print_header, print_kv

    print_header("OMEGA Status")

    kv: list[tuple[str, str]] = []

    # SQLite database (primary backend)
    db_path = OMEGA_DIR / "omega.db"
    if db_path.exists():
        import sqlite3

        size_mb = db_path.stat().st_size / (1024 * 1024)
        kv.append(("Backend", "SQLite"))
        kv.append(("Database", str(db_path)))
        kv.append(("Size", f"{size_mb:.2f} MB"))
        try:
            conn = sqlite3.connect(str(db_path))
            count = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
            kv.append(("Memories", str(count)))
            # Check sqlite-vec availability
            try:
                import sqlite_vec

                conn.enable_load_extension(True)
                sqlite_vec.load(conn)
                conn.enable_load_extension(False)
                kv.append(("Vector search", "enabled (sqlite-vec)"))
            except Exception:
                kv.append(("Vector search", "text-only fallback"))
            conn.close()
        except Exception as e:
            kv.append(("Error", str(e)))
    else:
        # Legacy JSONL store
        store_path = OMEGA_DIR / "store.jsonl"
        if store_path.exists():
            size_mb = store_path.stat().st_size / (1024 * 1024)
            with open(store_path) as f:
                line_count = sum(1 for _ in f)
            kv.append(("Backend", "JSONL (legacy)"))
            kv.append(("Store", str(store_path)))
            kv.append(("Memories", str(line_count)))
            kv.append(("Size", f"{size_mb:.2f} MB"))
            kv.append(("Tip", "Run 'omega migrate-db' to upgrade to SQLite"))
        else:
            kv.append(("Store", "not initialized"))
            kv.append(("Memories", "0"))

    # Model
    bge_path = BGE_MODEL_DIR / "model.onnx"
    minilm_path = MINILM_MODEL_DIR / "model.onnx"
    if bge_path.exists():
        model_mb = bge_path.stat().st_size / (1024 * 1024)
        kv.append(("Model", f"bge-small-en-v1.5 ONNX ({model_mb:.0f} MB)"))
    elif minilm_path.exists():
        model_mb = minilm_path.stat().st_size / (1024 * 1024)
        kv.append(("Model", f"all-MiniLM-L6-v2 ONNX ({model_mb:.0f} MB)"))
        kv.append(("Tip", "Run 'omega setup --download-model' to upgrade to bge-small-en-v1.5"))
    else:
        kv.append(("Model", "not downloaded"))
        kv.append(("Tip", "Run 'omega setup' to download"))

    # Legacy graphs (show if they still exist, suggest migration)
    graphs_dir = OMEGA_DIR / "graphs"
    if graphs_dir.exists():
        graph_files = list(graphs_dir.glob("*.json"))
        if graph_files:
            kv.append(("Legacy graphs", f"{len(graph_files)} files (run 'omega migrate-db' to convert)"))

    # Profile
    profile_path = OMEGA_DIR / "profile.json"
    if profile_path.exists():
        kv.append(("Profile", str(profile_path)))

    # Config
    config_path = OMEGA_DIR / "config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text())
        kv.append(("Version", config.get("version", "unknown")))

    print_kv(kv)

    # Cloud
    secrets_path = OMEGA_DIR / "secrets.json"
    if secrets_path.exists():
        cloud_kv = [("Cloud", "configured")]
        pull_marker = OMEGA_DIR / "last-cloud-pull"
        if pull_marker.exists():
            try:
                ts = pull_marker.read_text().strip()
                cloud_kv.append(("Last pull", ts))
            except Exception:
                pass
        push_marker = OMEGA_DIR / "last-cloud-push"
        if push_marker.exists():
            try:
                ts = push_marker.read_text().strip()
                cloud_kv.append(("Last push", ts))
            except Exception:
                pass
        print_kv(cloud_kv)
    else:
        print_kv([("Cloud", "not configured")])

    print()


def cmd_migrate(args):
    """Migrate data from MAGMA (~/.magma/) to OMEGA (~/.omega/). Non-destructive copy."""
    print("Migrating MAGMA data to OMEGA...")

    OMEGA_DIR.mkdir(parents=True, exist_ok=True, mode=0o700)
    (OMEGA_DIR / "graphs").mkdir(exist_ok=True, mode=0o700)

    copied = 0

    # Copy store.jsonl
    src = MAGMA_DIR / "store.jsonl"
    dst = OMEGA_DIR / "store.jsonl"
    if src.exists() and not dst.exists():
        shutil.copy2(src, dst)
        print(f"  Copied store.jsonl ({src.stat().st_size / 1024:.1f} KB)")
        copied += 1
    elif src.exists() and dst.exists():
        print(f"  Skipping store.jsonl (already exists in {OMEGA_DIR})")
    else:
        print(f"  No store.jsonl found at {src}")

    # Copy facts.jsonl
    src = MAGMA_DIR / "facts.jsonl"
    dst = OMEGA_DIR / "facts.jsonl"
    if src.exists() and not dst.exists():
        shutil.copy2(src, dst)
        print("  Copied facts.jsonl")
        copied += 1

    # Copy profile.json
    src = MAGMA_DIR / "profile.json"
    dst = OMEGA_DIR / "profile.json"
    if src.exists() and not dst.exists():
        shutil.copy2(src, dst)
        print("  Copied profile.json")
        copied += 1

    # Copy config.json (update storage_path)
    src = MAGMA_DIR / "config.json"
    dst = OMEGA_DIR / "config.json"
    if src.exists() and not dst.exists():
        config = json.loads(src.read_text())
        # Update paths
        for key in list(config.keys()):
            if isinstance(config[key], str):
                config[key] = config[key].replace(".magma", ".omega").replace("gnosis", "omega")
        dst.write_text(json.dumps(config, indent=2))
        print("  Copied config.json (paths updated)")
        copied += 1

    # Copy graph state files
    if MAGMA_GRAPHS.exists():
        for graph_file in MAGMA_GRAPHS.glob("*.json"):
            dst = OMEGA_DIR / "graphs" / graph_file.name
            if not dst.exists():
                shutil.copy2(graph_file, dst)
                print(f"  Copied graph: {graph_file.name}")
                copied += 1

    # Symlink ONNX model if available from gnosis
    gnosis_model = Path.home() / ".cache" / "gnosis" / "models" / "all-MiniLM-L6-v2-onnx"
    omega_model = OMEGA_CACHE / "models" / "all-MiniLM-L6-v2-onnx"
    if gnosis_model.exists() and not omega_model.exists():
        omega_model.parent.mkdir(parents=True, exist_ok=True)
        omega_model.symlink_to(gnosis_model)
        print(f"  Symlinked ONNX model from {gnosis_model}")
        copied += 1

    if copied > 0:
        print(f"\nMigration complete! Copied {copied} files.")
    else:
        print("\nNothing to migrate (all files already exist or no MAGMA data found).")
    print("Original MAGMA data is untouched.")

    # Auto-reingest into graph system
    store_path = OMEGA_DIR / "store.jsonl"
    if store_path.exists():
        print("\nIngesting store.jsonl into graph system...")
        cmd_reingest(args)


def cmd_reingest(args):
    """Reingest JSONL entries into the SQLite database."""
    store_path = OMEGA_DIR / "store.jsonl"
    pre_sqlite = OMEGA_DIR / "store.jsonl.pre-sqlite"
    # Check both current and backed-up JSONL
    if pre_sqlite.exists() and not store_path.exists():
        store_path = pre_sqlite
    if not store_path.exists():
        print(f"No JSONL store found at {OMEGA_DIR}")
        print("  Nothing to reingest (SQLite is the primary store now)")
        return

    from omega.bridge import reingest

    result = reingest(store_path=store_path)

    print("\nReingest complete:")
    print(f"  Ingested:   {result.get('ingested', 0)}")
    print(f"  Duplicates: {result.get('duplicates', 0)}")
    print(f"  Skipped:    {result.get('skipped', 0)}")
    print(f"  Errors:     {result.get('errors', 0)}")
    print(f"  Total:      {result.get('total', 0)}")

    from omega.bridge import status as omega_status

    s = omega_status()
    print(f"\nNode count: {s.get('node_count', 0)}")


def cmd_consolidate(args):
    """Run memory consolidation: deduplicate and prune old entries."""
    prune_days = getattr(args, "prune_days", 30)
    print(f"Running OMEGA consolidation (prune_days={prune_days})...")

    from omega.bridge import _get_store, deduplicate

    db = _get_store()
    node_count_before = db.node_count()
    print(f"  Nodes before: {node_count_before}")

    # Run deduplication via bridge
    result = deduplicate()
    merged = result.get("merged", 0) if isinstance(result, dict) else 0

    # Prune expired
    expired = db.cleanup_expired()

    # Evict old low-access entries if requested
    evicted = 0
    if prune_days > 0:
        evicted = db.evict_lru(count=0)  # 0 = only expired

    node_count_after = db.node_count()

    print("\nConsolidation complete:")
    print(f"  Duplicates merged: {merged}")
    print(f"  Expired pruned:    {expired}")
    print(f"  Evicted:           {evicted}")
    print(f"  Nodes after:       {node_count_after}")


def cmd_migrate_db(args):
    """Migrate from JSON graphs + JSONL to SQLite backend."""
    force = getattr(args, "force", False)
    from omega.migrate_to_sqlite import migrate

    report = migrate(force=force)
    if report.get("warnings"):
        for w in report["warnings"]:
            print(f"  WARNING: {w}")


def cmd_backup(args):
    """Back up omega.db to ~/.omega/backups/ with timestamp."""
    db_path = OMEGA_DIR / "omega.db"
    if not db_path.exists():
        print("No omega.db found — nothing to back up.")
        return

    backups_dir = OMEGA_DIR / "backups"
    backups_dir.mkdir(parents=True, exist_ok=True, mode=0o700)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    backup_path = backups_dir / f"omega-{timestamp}.db"

    import sqlite3
    from omega.crypto import secure_connect

    src = sqlite3.connect(str(db_path))
    dst = secure_connect(backup_path)
    src.backup(dst)
    dst.close()
    src.close()

    size_mb = backup_path.stat().st_size / (1024 * 1024)
    print(f"Backup saved: {backup_path} ({size_mb:.2f} MB)")

    # Rotate — keep only the 5 most recent backups
    backups = sorted(backups_dir.glob("omega-*.db"), key=lambda p: p.stat().st_mtime, reverse=True)
    for old in backups[5:]:
        old.unlink()
        print(f"  Rotated old backup: {old.name}")


def cmd_compact(args):
    """Cluster and summarize related memories to reduce noise."""
    event_type = getattr(args, "type", "lesson_learned")
    threshold = getattr(args, "threshold", 0.60)
    dry_run = getattr(args, "dry_run", False)

    print(f"Compacting {event_type} (threshold={threshold}, dry_run={dry_run})...")

    from omega.bridge import compact

    result = compact(
        event_type=event_type,
        similarity_threshold=threshold,
        dry_run=dry_run,
    )
    print(result)


def cmd_stats(args):
    """Show memory type distribution and health summary."""
    use_json = getattr(args, "json", False)

    from omega.bridge import type_stats, status as omega_status

    stats = type_stats()
    health = omega_status()

    if use_json:
        print(json.dumps({"types": stats, "health": health}, indent=2, default=str))
        return

    from omega.cli_ui import print_bar_chart, print_header, print_kv

    total = sum(stats.values())
    print_header("OMEGA Stats")
    print_kv(
        [
            ("Memories", str(total)),
            ("DB size", f"{health.get('db_size_mb', 0):.2f} MB"),
            ("Edges", str(health.get("edge_count", 0))),
            ("Backend", health.get("backend", "unknown")),
        ]
    )
    print()
    items = sorted(stats.items(), key=lambda x: -x[1])
    print_bar_chart(items, title="Type Distribution", total=total)


def cmd_activity(args):
    """Show recent session activity: sessions, tasks, insights, claims."""
    days = getattr(args, "days", 7)
    use_json = getattr(args, "json", False)

    from omega.bridge import get_activity_summary

    data = get_activity_summary(days=days)

    if use_json:
        print(json.dumps(data, indent=2, default=str))
        return

    from omega.cli_ui import print_header, print_section, print_table

    print_header(f"OMEGA Activity (last {days} days)")

    # Sessions
    print_section("Active Sessions")
    if data["sessions"]:
        rows = []
        for s in data["sessions"]:
            project = s.get("project") or ""
            rows.append(
                (
                    s.get("session_id") or "",
                    project.split("/")[-1] or project,
                    (s.get("task") or "")[:50],
                    (s.get("started_at") or "")[:19],
                    s.get("status") or "",
                )
            )
        print_table(
            None,
            ["Session", "Project", "Task", "Started", "Status"],
            rows,
            styles=["cyan", "bold", None, "dim", "green"],
        )
    else:
        print("  No active sessions")

    # Tasks
    print_section("Open Tasks")
    if data["tasks"]:
        rows = []
        for t in data["tasks"]:
            progress = f"{t.get('progress', 0)}%" if t.get("status") == "in_progress" else ""
            rows.append(
                (
                    str(t.get("id", "")),
                    t.get("title", "")[:50],
                    t.get("status", ""),
                    progress,
                    t.get("created_at", "")[:19],
                )
            )
        print_table(
            None,
            ["ID", "Title", "Status", "Progress", "Created"],
            rows,
            styles=["dim", "bold", "yellow", "cyan", "dim"],
        )
    else:
        print("  No open tasks")

    # Recent Insights
    print_section("Recent Insights")
    if data["insights"]:
        rows = []
        for i in data["insights"]:
            rows.append(
                (
                    i.get("type", ""),
                    i.get("preview", "")[:80],
                    i.get("created_at", "")[:19],
                    i.get("id", ""),
                )
            )
        print_table(None, ["Type", "Preview", "Created", "ID"], rows, styles=["bold", None, "dim", "dim"])
    else:
        print("  No recent insights")

    # Claims
    print_section("Active Claims")
    if data["claims"]:
        rows = []
        for c in data["claims"]:
            rows.append(
                (
                    c.get("type", ""),
                    c.get("path", ""),
                    c.get("session", ""),
                )
            )
        print_table(None, ["Type", "Path/Branch", "Session"], rows, styles=["bold", None, "dim"])
    else:
        print("  No active claims")


def _send_notification(text: str, context: str = None):
    """Send a macOS notification via osascript. Best-effort."""
    try:
        text_escaped = text.replace('"', '\\"')
        subtitle = ""
        if context:
            ctx_escaped = context[:80].replace('"', '\\"')
            subtitle = f' subtitle "{ctx_escaped}"'
        script = f'display notification "{text_escaped}" with title "OMEGA Reminder"{subtitle} sound name "Glass"'
        subprocess.run(
            ["osascript", "-e", script],
            capture_output=True,
            timeout=5,
        )
    except Exception:
        pass  # Best-effort


def cmd_remind(args):
    """Manage reminders: set, list, check, dismiss."""
    sub = getattr(args, "remind_command", None)

    if sub == "set":
        text = " ".join(args.text)
        duration = args.duration
        context = getattr(args, "context", None)
        if not text.strip():
            print("Usage: omega remind set <text> -d <duration>", file=sys.stderr)
            sys.exit(1)

        from omega.bridge import create_reminder

        try:
            result = create_reminder(text=text, duration=duration, context=context)
            print(f"Reminder set: {result['text']}")
            print(f"  Due at: {result['remind_at_local']}")
            print(f"  ID: {result['reminder_id']}")
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

    elif sub == "list":
        from omega.bridge import list_reminders

        status = getattr(args, "status", None)
        include_dismissed = status in ("dismissed", "all")
        reminders = list_reminders(status=status, include_dismissed=include_dismissed)

        if not reminders:
            print("No reminders found.")
            return

        print(f"Reminders ({len(reminders)} found):\n")
        for r in reminders:
            overdue = " [OVERDUE]" if r.get("is_overdue") else ""
            print(f"  [{r['status']}]{overdue} {r['text']}")
            print(f"    Due: {r['remind_at_local']} | Time: {r['time_until']}")
            if r.get("context"):
                print(f"    Context: {r['context'][:120]}")
            print(f"    ID: {r['id']}")

    elif sub == "check":
        from omega.bridge import get_due_reminders

        notify = getattr(args, "notify", False)
        due = get_due_reminders(mark_fired=True)

        if not due:
            print("No reminders due.")
            return

        for r in due:
            overdue = " [OVERDUE]" if r.get("is_overdue") else ""
            print(f"[REMINDER]{overdue} {r['text']}")
            if r.get("context"):
                print(f"  Context: {r['context'][:120]}")
            print(f"  ID: {r['id']}")

            if notify:
                _send_notification(r["text"], r.get("context"))

    elif sub == "dismiss":
        reminder_id = args.reminder_id
        from omega.bridge import dismiss_reminder

        result = dismiss_reminder(reminder_id)
        if result.get("success"):
            print(f"Dismissed: {result.get('text', reminder_id)}")
        else:
            print(f"Error: {result.get('error')}", file=sys.stderr)
            sys.exit(1)

    else:
        print("Usage: omega remind {set,list,check,dismiss}", file=sys.stderr)
        sys.exit(1)


def cmd_logs(args):
    """Show recent entries from ~/.omega/hooks.log."""
    hooks_log = OMEGA_DIR / "hooks.log"
    if not hooks_log.exists():
        print("No hooks.log found — no hook errors recorded.")
        return

    n = getattr(args, "lines", 50)
    lines = hooks_log.read_text().strip().split("\n")
    recent = lines[-n:] if len(lines) > n else lines
    print(f"--- Last {len(recent)} lines from {hooks_log} ---\n")
    for line in recent:
        print(line)


def cmd_validate(args):
    """Validate omega.db integrity: SQLite PRAGMA + FTS5 checks."""
    from omega.cli_ui import print_header, print_section, print_status_line, print_summary, print_table

    db_path = OMEGA_DIR / "omega.db"
    if not db_path.exists():
        print("No omega.db found.")
        return

    import sqlite3

    conn = sqlite3.connect(str(db_path))
    errors = 0

    print_header("OMEGA Validate")

    # SQLite integrity check
    print_section("SQLite Integrity")
    result = conn.execute("PRAGMA integrity_check").fetchone()[0]
    if result == "ok":
        print_status_line("ok", "PRAGMA integrity_check passed")
    else:
        errors += 1
        print_status_line("fail", result)

    # FTS5 integrity
    print_section("FTS5 Index")
    try:
        conn.execute("INSERT INTO memories_fts(memories_fts) VALUES('integrity-check')")
        print_status_line("ok", "FTS5 integrity check passed")
    except Exception as e:
        errors += 1
        print_status_line("fail", f"FTS5 integrity: {e}")
        if getattr(args, "repair", False):
            print("  Attempting rebuild...")
            try:
                conn.execute("INSERT INTO memories_fts(memories_fts) VALUES('rebuild')")
                conn.commit()
                print_status_line("ok", "FTS5 index rebuilt")
                errors -= 1
            except Exception as rebuild_err:
                print_status_line("fail", f"Rebuild failed: {rebuild_err}")

    # Row counts (allowlist — these names are used in f-string SQL)
    print_section("Table Counts")
    _VALID_TABLES = frozenset(
        [
            "memories",
            "edges",
            "entity_index",
            "coord_sessions",
            "coord_file_claims",
            "coord_branch_claims",
            "coord_intents",
            "coord_snapshots",
            "coord_tasks",
            "coord_audit",
        ]
    )
    table_rows = []
    for tbl in sorted(_VALID_TABLES):
        try:
            count = conn.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
            table_rows.append((tbl, str(count)))
        except Exception:
            pass  # Table may not exist
    print_table(None, ["Table", "Count"], table_rows)

    conn.close()
    print()
    print_summary(errors, 0)
    sys.exit(1 if errors > 0 else 0)


def cmd_serve(args):
    """Run the OMEGA MCP server (stdio mode)."""
    import asyncio
    from omega.server.mcp_server import main

    asyncio.run(main())


def cmd_doctor(args):
    """Verify OMEGA installation: import, model, database, MCP, hooks."""
    from omega.cli_ui import print_header, print_section, print_status_line, print_summary

    errors = 0
    warnings = 0

    def ok(msg):
        print_status_line("ok", msg)

    def fail(msg):
        nonlocal errors
        errors += 1
        print_status_line("fail", msg)

    def warn(msg):
        nonlocal warnings
        warnings += 1
        print_status_line("warn", msg)

    print_header("OMEGA Doctor")

    # 1. Package import
    print_section("Package Import")
    try:
        import omega

        ok(f"omega {omega.__version__} imported")
    except Exception as e:
        fail(f"Cannot import omega: {e}")
        print(f"\n{errors} error(s), {warnings} warning(s)")
        sys.exit(1)

    try:
        from omega.bridge import status as _s, auto_capture as _ac, query as _q  # noqa: F811,F401

        ok("omega.bridge imported (status, auto_capture, query)")
    except Exception as e:
        fail(f"Cannot import omega.bridge: {e}")

    try:
        from omega.server.handlers import HANDLERS

        ok(f"omega.server.handlers: {len(HANDLERS)} handlers registered")
    except Exception as e:
        fail(f"Cannot import handlers: {e}")

    try:
        from omega.server.tool_schemas import TOOL_SCHEMAS

        ok(f"omega.server.tool_schemas: {len(TOOL_SCHEMAS)} tools defined")
    except Exception as e:
        fail(f"Cannot import tool_schemas: {e}")

    # 2. ONNX model
    print_section("Embedding Model")
    bge_path = BGE_MODEL_DIR / "model.onnx"
    minilm_path = MINILM_MODEL_DIR / "model.onnx"
    if bge_path.exists():
        model_mb = bge_path.stat().st_size / (1024 * 1024)
        ok(f"bge-small-en-v1.5 model.onnx present ({model_mb:.0f} MB)")
        active_model_dir = BGE_MODEL_DIR
    elif minilm_path.exists():
        model_mb = minilm_path.stat().st_size / (1024 * 1024)
        ok(f"all-MiniLM-L6-v2 model.onnx present ({model_mb:.0f} MB)")
        warn("Using legacy model. Run 'omega setup --download-model' to upgrade to bge-small-en-v1.5")
        active_model_dir = MINILM_MODEL_DIR
    else:
        fail(f"model.onnx not found at {BGE_MODEL_DIR} or {MINILM_MODEL_DIR}")
        active_model_dir = BGE_MODEL_DIR

    tokenizer_path = active_model_dir / "tokenizer.json"
    if tokenizer_path.exists():
        ok("tokenizer.json present")
    else:
        fail(f"tokenizer.json not found at {active_model_dir}")

    try:
        from omega.graphs import generate_embedding, get_embedding_info

        info = get_embedding_info()
        if info.get("onnx_available"):
            ok("ONNX Runtime available")
        else:
            warn("ONNX Runtime not available, will use fallback")

        emb = generate_embedding("test embedding")
        if len(emb) == 384:
            ok(f"Embedding generation works (384-dim, backend={info.get('backend', 'unknown')})")
        else:
            fail(f"Embedding dimension wrong: {len(emb)} (expected 384)")
    except Exception as e:
        fail(f"Embedding generation failed: {e}")

    # 3. Database
    print_section("Database")
    db_path = OMEGA_DIR / "omega.db"
    if db_path.exists():
        size_mb = db_path.stat().st_size / (1024 * 1024)
        ok(f"omega.db exists ({size_mb:.2f} MB)")
    else:
        warn("omega.db not found (will be created on first use)")

    try:
        from omega.bridge import status as omega_status

        s = omega_status()
        # RSS will be high after loading ONNX model — only fail on actual DB issues
        db_ok = s.get("node_count", 0) >= 0 and s.get("backend") == "sqlite"
        if db_ok:
            ok(f"Database accessible: {s.get('node_count', 0)} memories, {s.get('db_size_mb', 0):.2f} MB")
        else:
            fail(f"Database issue: {s}")

        if s.get("vec_enabled"):
            ok("sqlite-vec enabled (vector search)")
        else:
            warn("sqlite-vec not available (text-only search)")
    except Exception as e:
        fail(f"Database check failed: {e}")

    # 4. MCP registration (client-specific)
    client = getattr(args, "client", None)
    check_claude = client == "claude-code" or shutil.which("claude")
    if check_claude:
        print_section("MCP Server (Claude Code)")
        try:
            result = subprocess.run(["claude", "mcp", "list"], capture_output=True, text=True, timeout=10)
            if "omega-memory" in result.stdout:
                ok("omega-memory registered in Claude Code")
            else:
                fail("omega-memory NOT registered in Claude Code")
                print("    Run: claude mcp add omega-memory -- python3 -m omega.server.mcp_server")
        except FileNotFoundError:
            warn("Claude Code CLI not found (cannot verify MCP registration)")
        except Exception as e:
            warn(f"MCP check failed: {e}")
    else:
        print_section("MCP Server")
        python_path = _resolve_python_path()
        ok(f"MCP server available: {python_path} -m omega.server.mcp_server")

    # 5. FTS5 health
    print_section("FTS5 Index")
    if db_path.exists():
        try:
            import sqlite3 as _sqlite3

            _conn = _sqlite3.connect(str(db_path))
            fts_count = _conn.execute("SELECT COUNT(*) FROM memories_fts").fetchone()[0]
            mem_count = _conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
            if fts_count > 0:
                ok(f"FTS5 index populated ({fts_count} entries, {mem_count} memories)")
                if abs(fts_count - mem_count) > mem_count * 0.1:
                    warn(f"FTS5 index drift: {fts_count} vs {mem_count} memories (>10% mismatch)")
            else:
                warn("FTS5 index empty (text search will use slower LIKE fallback)")
            # Integrity check
            try:
                _conn.execute("INSERT INTO memories_fts(memories_fts) VALUES('integrity-check')")
                ok("FTS5 integrity check passed")
            except Exception as fts_err:
                fail(f"FTS5 integrity check failed: {fts_err}")
                print("    Fix: INSERT INTO memories_fts(memories_fts) VALUES('rebuild')")
            _conn.close()
        except Exception as e:
            warn(f"FTS5 check skipped: {e}")

    # 5b. Vec index health
    print_section("Vector Index")
    if db_path.exists():
        try:
            import sqlite3 as _sqlite3

            _conn = _sqlite3.connect(str(db_path))
            try:
                import sqlite_vec

                _conn.enable_load_extension(True)
                sqlite_vec.load(_conn)
                _conn.enable_load_extension(False)
            except Exception:
                pass  # sqlite-vec may not be installed
            try:
                vec_count = _conn.execute("SELECT COUNT(*) FROM memories_vec").fetchone()[0]
                mem_count = _conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
                ok(f"Vec index: {vec_count} embeddings, {mem_count} memories")
                if vec_count > mem_count:
                    orphans = vec_count - mem_count
                    warn(f"Vec index has ~{orphans} potential orphaned embeddings (run 'omega consolidate' to clean)")
            except Exception as e:
                warn(f"Vec table not available: {e}")
            _conn.close()
        except Exception as e:
            warn(f"Vec check skipped: {e}")

    # 6. Coordination tables
    print_section("Coordination")
    if db_path.exists():
        try:
            import sqlite3 as _sqlite3

            _conn = _sqlite3.connect(str(db_path))
            coord_tables = [
                "coord_sessions",
                "coord_file_claims",
                "coord_branch_claims",
                "coord_intents",
                "coord_snapshots",
                "coord_tasks",
                "coord_audit",
            ]
            found = 0
            for tbl in coord_tables:
                row = _conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (tbl,)).fetchone()
                if row:
                    found += 1
            if found == len(coord_tables):
                ok(f"All {found} coordination tables present")
            elif found > 0:
                warn(f"Only {found}/{len(coord_tables)} coordination tables found")
            else:
                warn("No coordination tables (run any coordination tool to create them)")

            # Check stale sessions
            try:
                cutoff = (datetime.now(timezone.utc) - timedelta(seconds=360)).isoformat()
                stale = _conn.execute(
                    "SELECT COUNT(*) FROM coord_sessions WHERE last_heartbeat < ?", (cutoff,)
                ).fetchone()[0]
                if stale > 0:
                    warn(f"{stale} stale session(s) (heartbeat >360s ago)")
                else:
                    ok("No stale sessions")
            except Exception:
                pass  # coord_sessions may not exist yet

            _conn.close()
        except Exception as e:
            warn(f"Coordination check skipped: {e}")

    # 7. Memory quality
    print_section("Memory Quality")
    if db_path.exists():
        try:
            import sqlite3 as _sqlite3

            _conn = _sqlite3.connect(str(db_path))
            # Feedback stats
            rows = _conn.execute("SELECT metadata FROM memories WHERE metadata LIKE '%feedback_score%'").fetchall()
            if rows:
                scores = []
                flagged = 0
                for (meta_str,) in rows:
                    try:
                        meta = json.loads(meta_str)
                        scores.append(meta.get("feedback_score", 0))
                        if meta.get("flagged_for_review"):
                            flagged += 1
                    except Exception:
                        pass
                if scores:
                    avg = sum(scores) / len(scores)
                    ok(f"{len(scores)} memories with feedback (avg score: {avg:.2f})")
                    if flagged > 0:
                        warn(f"{flagged} memory(ies) flagged for review (score <= -3)")
            else:
                ok("No feedback signals recorded yet")
            _conn.close()
        except Exception as e:
            warn(f"Quality check skipped: {e}")

    # 8. Recent hook errors
    print_section("Hook Health")
    hooks_log = OMEGA_DIR / "hooks.log"
    if hooks_log.exists():
        try:
            lines = hooks_log.read_text().strip().split("\n")
            error_lines = [line for line in lines if line.startswith("[") and ": OK " not in line]
            if error_lines:
                recent = error_lines[-5:]
                warn(f"{len(error_lines)} hook error(s) in log, last {len(recent)}:")
                for line in recent:
                    print(f"    {line[:120]}")
            else:
                ok("No hook errors in log")
        except Exception as e:
            warn(f"Cannot read hooks.log: {e}")
    else:
        ok("No hooks.log (no errors recorded)")

    # 9. Hooks configuration (Claude Code-specific)
    check_hooks = client == "claude-code" or SETTINGS_JSON_PATH.exists()
    if check_hooks:
        print_section("Hooks (Claude Code)")
        if SETTINGS_JSON_PATH.exists():
            try:
                settings = json.loads(SETTINGS_JSON_PATH.read_text())
                hooks = settings.get("hooks", {})
                expected_events = ["SessionStart", "Stop", "PostToolUse"]
                for event in expected_events:
                    found = False
                    for entry in hooks.get(event, []):
                        for h in entry.get("hooks", []):
                            if "omega" in h.get("command", ""):
                                found = True
                                cmd_parts = h["command"].split()
                                if cmd_parts and not Path(cmd_parts[0]).exists():
                                    warn(f"{event} hook references {cmd_parts[0]} which doesn't exist")
                                break
                    if found:
                        ok(f"{event} hook configured")
                    else:
                        warn(f"{event} hook not configured")
            except Exception as e:
                warn(f"Cannot read settings.json: {e}")
        else:
            warn("settings.json not found (hooks not configured)")

    # 6. Python path
    print_section("Environment")
    python_path = _resolve_python_path()
    if Path(python_path).exists():
        ok(f"Python: {python_path}")
    else:
        fail(f"Python path does not exist: {python_path}")

    ok(f"OMEGA home: {OMEGA_DIR}")
    ok(f"Platform: {sys.platform}")

    # Summary
    print()
    print_summary(errors, warnings)
    sys.exit(1 if errors > 0 else 0)


def cmd_knowledge(args):
    """Knowledge base management."""
    try:
        from omega.knowledge.engine import scan_directory, list_documents, search_documents  # noqa: F401
    except ImportError:
        print("Knowledge base is a pro feature.")
        print("Learn more: https://omegamemory.com/pro")
        return

    subcmd = getattr(args, "kb_command", None)

    if subcmd == "scan":
        directory = args.dir
        result = scan_directory(directory)
        print(result)

    elif subcmd == "list":
        print(list_documents())

    elif subcmd == "search":
        query_text = " ".join(args.query)
        result = search_documents(query_text, limit=args.limit)
        print(result)

    elif subcmd == "sync-kb":
        from omega.knowledge.cloud_sync import sync_kb_queue
        result = sync_kb_queue(batch_size=args.batch_size)
        print(result)

    else:
        docs_dir = Path.home() / ".omega" / "documents"
        print("Usage: omega knowledge {scan|list|search}")
        print(f"\nDocuments folder: {docs_dir}")
        print("Drop PDF, markdown, or text files there for auto-ingestion.")
        print("Files are auto-scanned on each Claude Code session start.")


def cmd_cloud(args):
    """Cloud sync and Supabase management."""
    try:
        from omega.cloud.sync import get_sync  # noqa: F401
    except ImportError:
        print("Cloud sync is a pro feature.")
        print("Learn more: https://omegamemory.com/pro")
        return

    from omega.cli_ui import print_header

    subcmd = getattr(args, "cloud_command", None)

    if subcmd == "setup":
        url = args.url
        key = args.key
        service_key = args.service_key or ""
        if not url or not key:
            print("Usage: omega cloud setup --url <SUPABASE_URL> --key <ANON_KEY>")
            print("\nGet these from: Supabase Dashboard → Settings → API")
            return
        from omega.cloud.setup import setup_supabase

        result = setup_supabase(url, key, service_key)
        print(result)

    elif subcmd == "sync":
        print_header("Cloud Sync")
        try:
            sync = get_sync()
            results = sync.sync_all()
            for table, info in results.items():
                status = info.get("status", "unknown")
                synced = info.get("synced", 0)
                print(f"  {table}: {synced} synced ({status})")
        except Exception as e:
            print(f"  Sync failed: {e}")

    elif subcmd == "status":
        try:
            print(get_sync().status())
        except Exception as e:
            print(f"Cloud not configured: {e}")

    elif subcmd == "schema":
        from omega.cloud.setup import get_schema_sql

        print(get_schema_sql())

    elif subcmd == "verify":
        from omega.cloud.setup import verify_connection

        print(verify_connection())

    elif subcmd == "pull":
        print_header("Cloud Pull")
        try:
            sync = get_sync()
            results = sync.pull_all()
            for table, info in results.items():
                status = info.get("status", "unknown")
                pulled = info.get("pulled", 0)
                skipped = info.get("skipped", 0)
                print(f"  {table}: {pulled} pulled, {skipped} skipped ({status})")
        except Exception as e:
            print(f"  Pull failed: {e}")

    else:
        print("Usage: omega cloud {setup|sync|pull|status|schema|verify}")
        print("\nCloud sync enables mobile access to OMEGA memories via Supabase.")


def cmd_mobile(args):
    """Mobile access setup and mcp-proxy management."""
    try:
        from omega.cloud.sync import get_sync  # noqa: F401
    except ImportError:
        print("Mobile access is a pro feature.")
        print("Learn more: https://omegamemory.com/pro")
        return

    subcmd = getattr(args, "mobile_command", None)

    if subcmd == "setup":
        print("""
## OMEGA Mobile Access Setup

### Prerequisites
1. Install mcp-proxy: `pipx install mcp-proxy`
2. Install Tailscale: `brew install tailscale && tailscale up`

### Quick Start (4 steps)

1. Start OMEGA HTTP proxy:
   ```
   omega mobile serve
   ```

2. Expose via Tailscale:
   ```
   tailscale serve https / http://127.0.0.1:8089
   ```

3. Get your Tailscale hostname:
   ```
   tailscale status | head -1
   ```

4. Add to Claude mobile app:
   - Settings → MCP Servers → Add
   - URL: https://<your-tailscale-hostname>/mcp
   - All 70 OMEGA tools available from your phone!

### Security
- Tailscale uses WireGuard encryption (zero-trust mesh)
- Only your enrolled devices can connect
- No ports exposed to the public internet
- Encryption key stays on your Mac (profile decryption is local)

### Troubleshooting
- Verify: `curl http://127.0.0.1:8089/health`
- Tailscale: `tailscale status` (should show 'active')
- Logs: `omega logs -n 20`
""")

    elif subcmd == "serve":
        import subprocess
        import sys

        port = args.port
        host = args.host
        print(f"Starting OMEGA MCP proxy on {host}:{port}...")
        print(f"Connect via: http://{host}:{port}/mcp")
        print("Press Ctrl+C to stop.\n")

        try:
            subprocess.run(
                [
                    sys.executable, "-m", "mcp_proxy",
                    "--transport", "streamablehttp",
                    "--host", host,
                    "--port", str(port),
                    "--",
                    sys.executable, "-m", "omega.server.mcp_server",
                ],
                check=True,
            )
        except FileNotFoundError:
            print("Error: mcp-proxy not found. Install with: pipx install mcp-proxy")
        except KeyboardInterrupt:
            print("\nProxy stopped.")

    else:
        print("Usage: omega mobile {setup|serve}")
        print("\nMobile access via mcp-proxy + Tailscale.")


def main():
    parser = argparse.ArgumentParser(
        prog="omega",
        description="OMEGA — Persistent memory for AI coding agents",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- Memory commands ---
    query_parser = subparsers.add_parser("query", help="Search memories by semantic similarity or exact phrase")
    query_parser.add_argument("query_text", nargs="+", help="Search text")
    query_parser.add_argument("--exact", action="store_true", help="Use FTS5 exact phrase search instead of semantic")
    query_parser.add_argument("--limit", type=int, default=10, help="Max results (default: 10)")
    query_parser.add_argument("--json", action="store_true", help="Output as JSON")

    store_parser = subparsers.add_parser("store", help="Store a memory with a specified type")
    store_parser.add_argument("content", nargs="+", help="Memory content")
    store_parser.add_argument(
        "-t",
        "--type",
        default="memory",
        choices=["memory", "lesson", "decision", "error", "task", "preference"],
        help="Memory type (default: memory)",
    )

    remember_parser = subparsers.add_parser("remember", help="Store a permanent user preference")
    remember_parser.add_argument("text", nargs="+", help="Preference text")

    timeline_parser = subparsers.add_parser("timeline", help="Show memory timeline grouped by day")
    timeline_parser.add_argument("--days", type=int, default=7, help="Number of days to show (default: 7)")
    timeline_parser.add_argument("--json", action="store_true", help="Output as JSON")

    # --- Admin commands ---
    setup_parser = subparsers.add_parser("setup", help="Set up OMEGA: download model, initialize DB")
    setup_parser.add_argument(
        "--download-model",
        action="store_true",
        help="Download bge-small-en-v1.5 ONNX model (upgrade from all-MiniLM-L6-v2)",
    )
    setup_parser.add_argument(
        "--client", choices=["claude-code"], help="Configure a specific client (MCP registration, hooks)"
    )

    subparsers.add_parser("status", help="Show memory count, store size, model status")

    doctor_parser = subparsers.add_parser("doctor", help="Verify installation: import, model, database")
    doctor_parser.add_argument("--client", choices=["claude-code"], help="Include client-specific checks (MCP, hooks)")

    subparsers.add_parser("migrate", help="Copy MAGMA data to OMEGA (non-destructive)")
    migrate_db_parser = subparsers.add_parser("migrate-db", help="Migrate JSON graphs to SQLite backend")
    migrate_db_parser.add_argument("--force", action="store_true", help="Overwrite existing SQLite database")
    subparsers.add_parser("reingest", help="Load store.jsonl entries into graph system")
    consolidate_parser = subparsers.add_parser("consolidate", help="Deduplicate, prune, and optimize memory")
    consolidate_parser.add_argument(
        "--prune-days", type=int, default=30, help="Prune entries older than N days with 0 access (default: 30)"
    )
    subparsers.add_parser("backup", help="Back up omega.db to ~/.omega/backups/ (keeps last 5)")
    compact_parser = subparsers.add_parser("compact", help="Cluster and summarize related memories")
    compact_parser.add_argument(
        "-t",
        "--type",
        default="lesson_learned",
        choices=["lesson_learned", "decision", "error_pattern", "task_completion"],
        help="Event type to compact (default: lesson_learned)",
    )
    compact_parser.add_argument("--threshold", type=float, default=0.60, help="Similarity threshold (default: 0.60)")
    compact_parser.add_argument(
        "--dry-run", action="store_true", help="Show what would be compacted without changing data"
    )
    stats_parser = subparsers.add_parser("stats", help="Show memory type distribution and health summary")
    stats_parser.add_argument("--json", action="store_true", help="Output as JSON")
    activity_parser = subparsers.add_parser("activity", help="Show recent session activity overview")
    activity_parser.add_argument("--days", type=int, default=7, help="Number of days to show (default: 7)")
    activity_parser.add_argument("--json", action="store_true", help="Output as JSON")
    logs_parser = subparsers.add_parser("logs", help="Show recent hook errors from hooks.log")
    logs_parser.add_argument("-n", "--lines", type=int, default=50, help="Number of lines to show (default: 50)")
    validate_parser = subparsers.add_parser("validate", help="Validate omega.db integrity (SQLite + FTS5)")
    validate_parser.add_argument("--repair", action="store_true", help="Attempt to repair FTS5 index if corrupted")
    subparsers.add_parser("serve", help="Run MCP server (stdio mode)")

    # --- Reminder commands (experimental) ---
    remind_parser = subparsers.add_parser("remind", help="Manage time-based reminders (experimental)")
    remind_sub = remind_parser.add_subparsers(dest="remind_command", help="Reminder subcommands")

    remind_set_parser = remind_sub.add_parser("set", help="Set a new reminder")
    remind_set_parser.add_argument("text", nargs="+", help="Reminder text")
    remind_set_parser.add_argument("-d", "--duration", required=True, help="Duration: 1h, 30m, 2d, 1w, 1d12h")
    remind_set_parser.add_argument("--context", help="Optional context for the reminder")

    remind_list_parser = remind_sub.add_parser("list", help="List reminders")
    remind_list_parser.add_argument(
        "--status",
        choices=["pending", "fired", "dismissed", "all"],
        help="Filter by status (default: pending + fired)",
    )

    remind_check_parser = remind_sub.add_parser("check", help="Check for due reminders")
    remind_check_parser.add_argument("--notify", action="store_true", help="Send macOS notification for due reminders")

    remind_dismiss_parser = remind_sub.add_parser("dismiss", help="Dismiss a reminder")
    remind_dismiss_parser.add_argument("reminder_id", help="Reminder ID to dismiss")

    # --- Knowledge commands ---
    knowledge_parser = subparsers.add_parser("knowledge", aliases=["kb"], help="Knowledge base management")
    knowledge_sub = knowledge_parser.add_subparsers(dest="kb_command", help="Knowledge subcommands")
    scan_parser = knowledge_sub.add_parser("scan", help="Scan documents folder for new/changed files")
    scan_parser.add_argument("--dir", help="Custom directory to scan (default: ~/.omega/documents/)")
    knowledge_sub.add_parser("list", help="List all ingested documents")
    knowledge_search_parser = knowledge_sub.add_parser("search", help="Search ingested documents")
    knowledge_search_parser.add_argument("query", nargs="+", help="Search query")
    knowledge_search_parser.add_argument("--limit", type=int, default=5, help="Max results (default: 5)")
    sync_kb_parser = knowledge_sub.add_parser("sync-kb", help="Sync pending files from cloud KB queue")
    sync_kb_parser.add_argument("--batch-size", type=int, default=10, help="Max items to process (default: 10)")

    # --- Cloud commands ---
    cloud_parser = subparsers.add_parser("cloud", help="Cloud sync and mobile access")
    cloud_sub = cloud_parser.add_subparsers(dest="cloud_command", help="Cloud subcommands")

    cloud_setup_parser = cloud_sub.add_parser("setup", help="Configure Supabase connection")
    cloud_setup_parser.add_argument("--url", help="Supabase project URL")
    cloud_setup_parser.add_argument("--key", help="Supabase anon key")
    cloud_setup_parser.add_argument("--service-key", help="Supabase service role key (optional)")

    cloud_sub.add_parser("sync", help="Sync local data to Supabase cloud")
    cloud_sub.add_parser("status", help="Show cloud sync status")
    cloud_sub.add_parser("schema", help="Print Supabase SQL schema")
    cloud_sub.add_parser("verify", help="Verify Supabase connection")
    cloud_sub.add_parser("pull", help="Pull memories and documents from Supabase cloud")

    # --- Mobile commands ---
    mobile_parser = subparsers.add_parser("mobile", help="Mobile access via mcp-proxy + Tailscale")
    mobile_sub = mobile_parser.add_subparsers(dest="mobile_command", help="Mobile subcommands")
    mobile_sub.add_parser("setup", help="Print setup instructions for mobile access")
    mobile_serve_parser = mobile_sub.add_parser("serve", help="Start mcp-proxy HTTP server for mobile access")
    mobile_serve_parser.add_argument("--port", type=int, default=8089, help="HTTP port (default: 8089)")
    mobile_serve_parser.add_argument("--host", default="127.0.0.1", help="Bind address (default: 127.0.0.1)")

    args = parser.parse_args()

    commands = {
        "query": cmd_query,
        "store": cmd_store,
        "remember": cmd_remember,
        "timeline": cmd_timeline,
        "setup": cmd_setup,
        "status": cmd_status,
        "doctor": cmd_doctor,
        "migrate": cmd_migrate,
        "migrate-db": cmd_migrate_db,
        "reingest": cmd_reingest,
        "consolidate": cmd_consolidate,
        "backup": cmd_backup,
        "compact": cmd_compact,
        "stats": cmd_stats,
        "activity": cmd_activity,
        "logs": cmd_logs,
        "validate": cmd_validate,
        "serve": cmd_serve,
        "remind": cmd_remind,
        "knowledge": cmd_knowledge,
        "kb": cmd_knowledge,
        "cloud": cmd_cloud,
        "mobile": cmd_mobile,
    }

    # Wire plugin CLI commands (omega-pro, etc.)
    try:
        from omega.plugins import discover_plugins
        for plugin in discover_plugins():
            for cmd_name, setup_func in getattr(plugin, "CLI_COMMANDS", []):
                if cmd_name not in commands:
                    try:
                        setup_func(subparsers)
                        commands[cmd_name] = getattr(plugin, f"cmd_{cmd_name}", None)
                    except Exception as e:
                        print(f"Warning: plugin CLI command '{cmd_name}' failed: {e}", file=sys.stderr)
    except Exception:
        pass  # Plugins unavailable -- core CLI still works

    if args.command in commands:
        commands[args.command](args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
