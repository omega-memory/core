"""Tests for OMEGA Context Virtualization — checkpoint + resume tools."""

import pytest

from omega.server.handlers import HANDLERS


# ============================================================================
# Fixture: reset bridge singleton between tests
# ============================================================================


@pytest.fixture(autouse=True)
def _reset_bridge(tmp_omega_dir):
    """Reset the bridge singleton so each test gets a fresh store."""
    from omega.bridge import reset_memory

    reset_memory()
    yield
    reset_memory()


# ============================================================================
# omega_checkpoint — basic storage
# ============================================================================


@pytest.mark.asyncio
async def test_checkpoint_basic():
    """Store a basic checkpoint with title and progress."""
    result = await HANDLERS["omega_checkpoint"](
        {
            "task_title": "Frontend redesign Phase 2",
            "progress": "Completed header, working on sidebar",
        }
    )
    assert not result.get("isError"), result
    text = result["content"][0]["text"]
    assert "Checkpoint #1 saved" in text
    assert "Frontend redesign Phase 2" in text


@pytest.mark.asyncio
async def test_checkpoint_missing_required_fields():
    """Should fail when missing task_title or progress."""
    result = await HANDLERS["omega_checkpoint"]({"task_title": "Test"})
    assert result.get("isError")

    result = await HANDLERS["omega_checkpoint"]({"progress": "working"})
    assert result.get("isError")

    result = await HANDLERS["omega_checkpoint"]({})
    assert result.get("isError")


@pytest.mark.asyncio
async def test_checkpoint_full_content():
    """Store a checkpoint with all fields and verify content format."""
    result = await HANDLERS["omega_checkpoint"](
        {
            "task_title": "API migration",
            "plan": "Migrate REST endpoints to GraphQL",
            "progress": "3 of 10 endpoints migrated",
            "files_touched": {
                "src/api/users.ts": "Converted to GraphQL resolver",
                "src/schema.graphql": "Added User type",
            },
            "decisions": [
                "Using Apollo Server over express-graphql",
                "Keeping REST for auth endpoints",
            ],
            "key_context": "Auth tokens use JWT with RS256. Base schema in schema.graphql.",
            "next_steps": "Migrate orders endpoint next. Follow pattern in users.ts.",
            "session_id": "test-session-123",
            "project": "/test/project",
        }
    )
    assert not result.get("isError"), result
    text = result["content"][0]["text"]
    assert "Checkpoint #1 saved" in text
    assert "API migration" in text


@pytest.mark.asyncio
async def test_checkpoint_increments_number():
    """Sequential checkpoints should increment the checkpoint number."""
    await HANDLERS["omega_checkpoint"](
        {
            "task_title": "Dashboard rebuild",
            "progress": "Phase 1 complete",
        }
    )
    result = await HANDLERS["omega_checkpoint"](
        {
            "task_title": "Dashboard rebuild",
            "progress": "Phase 2 complete",
        }
    )
    assert not result.get("isError"), result
    text = result["content"][0]["text"]
    assert "Checkpoint #2 saved" in text


# ============================================================================
# omega_resume_task — retrieval
# ============================================================================


@pytest.mark.asyncio
async def test_resume_no_checkpoints():
    """Should return helpful message when no checkpoints exist."""
    result = await HANDLERS["omega_resume_task"]({"task_title": "nonexistent task"})
    assert not result.get("isError")
    text = result["content"][0]["text"]
    assert "No checkpoints found" in text


@pytest.mark.asyncio
async def test_resume_full_verbosity():
    """Store a checkpoint then resume with full verbosity."""
    await HANDLERS["omega_checkpoint"](
        {
            "task_title": "Widget refactor",
            "plan": "Refactor Widget component tree",
            "progress": "Split WidgetContainer done, WidgetList in progress",
            "files_touched": {"src/Widget.tsx": "Split into container + list"},
            "decisions": ["Using compound component pattern"],
            "key_context": "Widget uses React context for theme. Provider at App level.",
            "next_steps": "Implement WidgetItem with drag-and-drop support",
        }
    )

    result = await HANDLERS["omega_resume_task"]({"task_title": "Widget refactor", "verbosity": "full"})
    assert not result.get("isError"), result
    text = result["content"][0]["text"]
    assert "Task Resume" in text
    assert "Widget refactor" in text
    assert "Progress" in text


@pytest.mark.asyncio
async def test_resume_summary_verbosity():
    """Resume with summary verbosity should include plan + progress + next steps."""
    await HANDLERS["omega_checkpoint"](
        {
            "task_title": "Auth system upgrade",
            "plan": "Move from session to JWT auth",
            "progress": "JWT generation done, middleware pending",
            "next_steps": "Add JWT middleware to protected routes",
        }
    )

    result = await HANDLERS["omega_resume_task"]({"task_title": "Auth system", "verbosity": "summary"})
    assert not result.get("isError"), result
    text = result["content"][0]["text"]
    assert "Task Resume" in text
    assert "Plan" in text or "Progress" in text


@pytest.mark.asyncio
async def test_resume_minimal_verbosity():
    """Resume with minimal verbosity should only show next steps."""
    await HANDLERS["omega_checkpoint"](
        {
            "task_title": "Perf optimization",
            "progress": "Profiled hot paths, fixed 2 of 5",
            "next_steps": "Fix N+1 query in UserList component",
        }
    )

    result = await HANDLERS["omega_resume_task"]({"task_title": "Perf optimization", "verbosity": "minimal"})
    assert not result.get("isError"), result
    text = result["content"][0]["text"]
    assert "Next Steps" in text


@pytest.mark.asyncio
async def test_resume_multiple_checkpoints():
    """Resume with limit > 1 should return multiple checkpoints."""
    for i in range(3):
        await HANDLERS["omega_checkpoint"](
            {
                "task_title": "Multi-step task",
                "progress": f"Step {i + 1} complete",
            }
        )

    result = await HANDLERS["omega_resume_task"]({"task_title": "Multi-step task", "limit": 3})
    assert not result.get("isError"), result
    text = result["content"][0]["text"]
    assert "checkpoint(s) found" in text


@pytest.mark.asyncio
async def test_resume_empty_title_returns_any():
    """Resume with no task_title should return any recent checkpoint."""
    await HANDLERS["omega_checkpoint"](
        {
            "task_title": "Some task",
            "progress": "In progress",
        }
    )

    result = await HANDLERS["omega_resume_task"]({})
    assert not result.get("isError"), result
    text = result["content"][0]["text"]
    assert "Task Resume" in text


# ============================================================================
# Round-trip: checkpoint → resume data integrity
# ============================================================================


@pytest.mark.asyncio
async def test_checkpoint_resume_roundtrip():
    """Full round-trip: checkpoint all fields, resume and verify data."""
    await HANDLERS["omega_checkpoint"](
        {
            "task_title": "E2E roundtrip test",
            "plan": "Verify data survives checkpoint/resume cycle",
            "progress": "Writing checkpoint with all fields",
            "files_touched": {
                "tests/test_checkpoint.py": "Added roundtrip test",
            },
            "decisions": ["Using structured JSON in metadata for reliable parsing"],
            "key_context": "Checkpoint data stored as both formatted content and structured metadata",
            "next_steps": "Verify resume returns all fields correctly",
        }
    )

    result = await HANDLERS["omega_resume_task"]({"task_title": "E2E roundtrip test", "verbosity": "full"})
    assert not result.get("isError"), result
    text = result["content"][0]["text"]
    assert "E2E roundtrip test" in text
    assert "Plan" in text or "Progress" in text


# ============================================================================
# TTL / Priority / Weight constants
# ============================================================================


def test_checkpoint_ttl_configured():
    """Checkpoint event type should have a 7-day TTL."""
    from omega.types import EVENT_TYPE_TTL

    assert EVENT_TYPE_TTL.get("checkpoint") == 604800


def test_checkpoint_type_weight():
    """Checkpoint type weight should be highest (2.5)."""
    from omega.sqlite_store import SQLiteStore

    assert SQLiteStore._TYPE_WEIGHTS.get("checkpoint") == 2.5


def test_checkpoint_default_priority():
    """Checkpoint default priority should be 5 (highest)."""
    from omega.sqlite_store import SQLiteStore

    assert SQLiteStore._DEFAULT_PRIORITY.get("checkpoint") == 5


def test_checkpoint_event_type_constant():
    """AutoCaptureEventType should have CHECKPOINT constant."""
    from omega.types import AutoCaptureEventType

    assert AutoCaptureEventType.CHECKPOINT == "checkpoint"


# ============================================================================
# Schema validation
# ============================================================================


def test_checkpoint_schema_exists():
    """omega_checkpoint should be in TOOL_SCHEMAS."""
    from omega.server.tool_schemas import TOOL_SCHEMAS

    names = [s["name"] for s in TOOL_SCHEMAS]
    assert "omega_checkpoint" in names


def test_resume_task_schema_exists():
    """omega_resume_task should be in TOOL_SCHEMAS."""
    from omega.server.tool_schemas import TOOL_SCHEMAS

    names = [s["name"] for s in TOOL_SCHEMAS]
    assert "omega_resume_task" in names


def test_checkpoint_handler_registered():
    """omega_checkpoint should be in HANDLERS."""
    assert "omega_checkpoint" in HANDLERS


def test_resume_task_handler_registered():
    """omega_resume_task should be in HANDLERS."""
    assert "omega_resume_task" in HANDLERS


def test_checkpoint_dedup_threshold():
    """Checkpoint dedup threshold should be 0.90."""
    from omega.bridge import DEDUP_THRESHOLDS

    assert DEDUP_THRESHOLDS.get("checkpoint") == 0.90


# ============================================================================
# Hook: _session_resume checkpoint surfacing
# ============================================================================


@pytest.mark.asyncio
async def test_session_resume_surfaces_checkpoint():
    """_session_resume should include [CHECKPOINT] when checkpoints exist."""
    # Store a checkpoint first
    await HANDLERS["omega_checkpoint"](
        {
            "task_title": "Hook surfacing test",
            "progress": "Testing hook integration",
            "next_steps": "Verify [CHECKPOINT] appears",
            "project": "/test/project",
        }
    )

    # Call _session_resume
    from omega.server.hook_server import _session_resume

    class FakeMgr:
        def recover_session(self, project):
            return []

    lines = _session_resume("test-session", "/test/project", FakeMgr())
    combined = "\n".join(lines)
    assert "[CHECKPOINT]" in combined
    assert "Hook surfacing test" in combined
