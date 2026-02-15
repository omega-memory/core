"""
OMEGA Protocol — JIT coordination instructions served dynamically.

Instead of static CLAUDE.md rules loaded every turn, this module provides
context-sensitive protocol sections that are served on-demand via the
omega_protocol() MCP tool.

Architecture:
- Base protocol sections live here as versioned, structured data
- OMEGA memories tagged event_type="protocol" augment with learned lessons
- The get_protocol() function assembles the right sections based on context
"""

import logging
from typing import Dict, List, Optional

logger = logging.getLogger("omega.protocol")

# Protocol version — bump when sections change materially
PROTOCOL_VERSION = "1.0.0"

# ---------------------------------------------------------------------------
# Protocol Sections — each is a (title, content) pair
# ---------------------------------------------------------------------------

SECTIONS: Dict[str, Dict[str, str]] = {
    "memory": {
        "title": "Memory Usage",
        "content": """\
- `[MEMORY]` blocks from hooks = ground truth, use immediately
- **Before** non-trivial tasks: `omega_query()` for prior context, decisions, gotchas
- **After** completing tasks: `omega_store(content, "decision"|"lesson_learned")` for key outcomes
- **On errors**: check OMEGA for prior solutions before debugging from scratch
- **User says "remember"**: `omega_store(text, "user_preference")`
- When asked about preferences/history: query OMEGA FIRST, don't say "I don't know"
- **Feedback loop**: After `omega_query` surfaces memories with IDs, use `omega_feedback(memory_id, "helpful"|"unhelpful"|"outdated")` to train the ranker
- **Graph intelligence**: Use `omega_traverse(memory_id, max_hops=2)` to walk relationship graphs. Use `omega_similar(memory_id)` to discover related clusters
- **Cross-session lessons**: At session start, call `omega_lessons(task="<description>")` for ranked lessons from prior sessions
- **Memory editing**: Prefer `omega_edit_memory(memory_id, new_content)` over delete+recreate""",
    },
    "coordination": {
        "title": "Multi-Agent Coordination",
        "content": """\
- `[HANDOFF]` blocks = predecessor's work summary, continue where they left off
- `[INBOX]` alerts = peer messages waiting, check `omega_inbox` for details
- `[DEADLOCK]` alerts = circular wait, release a file with `omega_file_release`
- `[COORD]` peer roster = ground truth on active work — don't overlap
- Before editing shared files: `omega_file_check(file_path=...)` for conflicts
- Multi-step work: `omega_task_create` with `omega_task_deps` for cross-agent visibility
- After significant work: `omega_task_complete(task_id=..., result="summary")`
- If task cannot be completed: `omega_task_fail(task_id, reason)` — never silently abandon
- Before multi-file changes: `omega_intent_announce(description=..., target_files=[...])`
- Before git branch operations: `omega_branch_check` then `omega_branch_claim`
- After merging/abandoning: `omega_branch_release` to free the claim
- **Audit trail**: Use `omega_audit` for debugging coordination issues
- **Session safety**: `omega_session_snapshot` before risky ops; `omega_session_recover` after crashes""",
    },
    "coordination_gate": {
        "title": "Coordination Gate (Risk-Tiered)",
        "content": """\
| Risk | Actions | Gate |
|------|---------|------|
| **HIGH** | Deploy, force-push, delete branch, rm -rf | Full gate (all 3 steps below) |
| **MEDIUM** | git commit/push, create branch, install deps | `omega_coord_status` only |
| **LOW** | Edit claimed files, run tests, read | No gate — hooks handle it |

### HIGH-risk gate (all 3 required):
1. `omega_query(event_type="decision", query="<target area>")` — check prior decisions
2. `omega_coord_status` — check peer activity and claimed files
3. `git log --oneline -10 -- <target_dir>` — verify recent changes

### MEDIUM-risk gate:
1. `omega_coord_status` — check peer activity. Warn on overlaps, don't block.

"Just deploying" and "routine task" are NOT reasons to skip HIGH-tier checks.""",
    },
    "teamwork": {
        "title": "Proactive Teamwork",
        "content": """\
### Task Completion Ritual (every task)
1. Check `omega_tasks_list` — does your output unblock a peer?
2. If yes: `omega_send_message` them with what they need
3. Leave a handoff: `omega_store("Completed <task>. Output: <what>. Gotchas: <what>.", "decision")`
4. Pre-stage next step: `omega_task_create` if follow-up is clear

### Mid-Task Awareness (every commit or 15+ min)
1. `omega_coord_status` — peer intersecting your work?
2. `omega_intent_check` — file list conflict with new intent?
3. Discoveries affecting peers → `omega_send_message` NOW

### Conflict Resolution
1. Later intent yields — earlier claim has priority
2. Propose a split via `omega_send_message` with concrete division
3. One exchange max — escalate to user if unresolved

### Pipeline Thinking
- Ask: "What happens after I finish? Who's waiting on me?"
- When blocked: check `omega_tasks_list` for unblocked work
- When idle: `omega_task_next` — pick up work, don't ask permission""",
    },
    "context": {
        "title": "Context Management",
        "content": """\
- When context window is getting full (>70%): `omega_checkpoint` to save task state
- When starting a session for an ongoing task: `omega_resume_task` first
- When `[CHECKPOINT]` appears at session start: offer to resume with `omega_resume_task`
- Checkpoints save: plan, progress, files changed, decisions, key context, next steps""",
    },
    "reminders": {
        "title": "Reminders",
        "content": """\
- When user says "remind me" or task has a future deadline: `omega_remind(text, duration)`
- Duration examples: '1h', '30m', '2d', '1w', '1d12h'
- At session start: check `omega_remind_list` for pending/fired reminders
- After acknowledging a reminder: `omega_remind_dismiss(reminder_id)`""",
    },
    "diagnostics": {
        "title": "Diagnostics & Maintenance",
        "content": """\
- Health/status/audit: `omega_health`, `omega_type_stats`, `omega_weekly_digest`, `omega_timeline`, `omega_forgetting_log`
- Before risky bulk operations: `omega_backup(filepath="~/.omega/backups/omega-<date>.json")`
- Periodic maintenance: `omega_compact`, `omega_consolidate`""",
    },
    "entity": {
        "title": "Entity & Knowledge",
        "content": """\
- **People/orgs**: Use entity tools (`omega_entity_create/get/list/update/relationships/tree`) with `entity_id` scoping
- **Documents**: `omega_search_documents(query)` before web search. Ingest: `omega_ingest_document(path)`
- **Profile data**: `omega_profile_set/get/search/list` for structured user data. Prefer profile tools over flat memories""",
    },
    "heuristics": {
        "title": "Decision Heuristics",
        "content": """\
- **Reversibility test**: Reversible → proceed. Irreversible → ask first.
- **Friction is signal**: Harder than expected? Investigate — usually means missing context.
- **Learn, don't just complete**: On mistakes, `omega_store` the lesson before moving on.
- **Push back from care**: If user's approach will cause problems, say so directly.
- **When in doubt, narrow scope**: Do less correctly rather than more with assumptions.

### Anti-Rationalization
| Thought | Do instead |
|---|---|
| "Just deploying" | Run the coordination gate |
| "This is routine" | Run the coordination gate |
| "I'll check after" | Check before — that's the point |
| "No one else is working" | Verify with `omega_coord_status` |
| "I know this area" | `omega_query` for decisions you missed |
| "It's a small change" | Small shared-file changes cause the biggest conflicts |""",
    },
    "git": {
        "title": "Git Rules",
        "content": """\
- Commit only files YOU modified. `git add <files>` — never `git add .`
- After every commit: `omega_store("Committed <hash>: <message>. Files: <list>", "decision")`
- Before "what's next": `omega_coord_status` + `omega_git_events` + `omega_query(event_type="decision")`""",
    },
    "what_next": {
        "title": "What's Next Protocol",
        "content": """\
Before recommending work or answering "what's next":
1. `omega_coord_status` — check active peers
2. `omega_git_events` — check recent commits
3. `omega_query(event_type="decision")` — check recent decisions
4. `ps aux | grep -E "python.*(benchmark|harness)"` — check running processes
Never advise work that's already done, overlaps a peer, or ignores running processes.""",
    },
}

# Section groups — named bundles for common scenarios
SECTION_GROUPS: Dict[str, List[str]] = {
    "solo": ["memory", "context", "reminders", "heuristics", "git"],
    "multi_agent": [
        "memory", "coordination", "coordination_gate", "teamwork",
        "context", "reminders", "heuristics", "git", "what_next",
    ],
    "full": list(SECTIONS.keys()),
    "minimal": ["memory", "context", "git"],
}


def get_protocol(
    section: Optional[str] = None,
    project: Optional[str] = None,
    include_lessons: bool = True,
    peer_count: int = 0,
) -> str:
    """Assemble the protocol playbook dynamically.

    Args:
        section: Specific section name, group name, or None for auto-detect.
        project: Current project path for context-sensitive rules.
        include_lessons: Whether to append relevant lessons from OMEGA.
        peer_count: Number of active peers (0 = solo mode).

    Returns:
        Formatted protocol text ready for agent consumption.
    """
    # Determine which sections to include
    if section and section in SECTIONS:
        # Single section requested
        selected = [section]
    elif section and section in SECTION_GROUPS:
        # Named group requested
        selected = SECTION_GROUPS[section]
    elif section == "all" or section == "full":
        selected = list(SECTIONS.keys())
    elif peer_count > 0:
        # Multi-agent mode — include coordination sections
        selected = SECTION_GROUPS["multi_agent"]
    else:
        # Solo mode — skip coordination overhead
        selected = SECTION_GROUPS["solo"]

    # Build output
    lines = [f"# OMEGA Protocol v{PROTOCOL_VERSION}\n"]

    if peer_count > 0:
        lines.append(f"_Mode: multi-agent ({peer_count} peer{'s' if peer_count != 1 else ''} active)_\n")
    else:
        lines.append("_Mode: solo_\n")

    for key in selected:
        sec = SECTIONS.get(key)
        if sec:
            lines.append(f"## {sec['title']}")
            lines.append(sec["content"])
            lines.append("")

    # Append relevant lessons from OMEGA if requested
    if include_lessons:
        lessons_text = _get_protocol_lessons(project)
        if lessons_text:
            lines.append("## Learned Protocol Lessons")
            lines.append(lessons_text)
            lines.append("")

    return "\n".join(lines)


def _get_protocol_lessons(project: Optional[str] = None) -> str:
    """Fetch relevant lessons about coordination/protocol from OMEGA memory."""
    try:
        from omega.bridge import query_structured

        results = query_structured(
            query_text="coordination protocol gate deployment gotcha",
            limit=5,
            event_type="lesson_learned",
        )
        if not results:
            return ""

        items = []
        for r in results[:5]:
            content = r.get("content", "")[:200]
            items.append(f"- {content}")
        return "\n".join(items)
    except Exception as e:
        logger.debug("Protocol lessons fetch failed: %s", e)
        return ""


def list_sections() -> List[Dict[str, str]]:
    """List all available protocol sections with titles."""
    return [
        {"key": key, "title": sec["title"]}
        for key, sec in SECTIONS.items()
    ]
