"""Tests for OMEGA Reminder System (experimental)."""

from datetime import datetime, timedelta, timezone

import pytest

from omega.bridge import (
    create_reminder,
    dismiss_reminder,
    get_due_reminders,
    list_reminders,
    parse_duration,
)
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
# Duration Parsing
# ============================================================================


class TestParseDuration:
    def test_minutes(self):
        assert parse_duration("30m") == timedelta(minutes=30)

    def test_hours(self):
        assert parse_duration("2h") == timedelta(hours=2)

    def test_days(self):
        assert parse_duration("3d") == timedelta(days=3)

    def test_weeks(self):
        assert parse_duration("1w") == timedelta(weeks=1)

    def test_compound(self):
        assert parse_duration("1d12h") == timedelta(days=1, hours=12)

    def test_verbose_hours(self):
        assert parse_duration("2 hours") == timedelta(hours=2)

    def test_verbose_minutes(self):
        assert parse_duration("30 minutes") == timedelta(minutes=30)

    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            parse_duration("abc")

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            parse_duration("")

    def test_zero_raises(self):
        with pytest.raises(ValueError):
            parse_duration("0m")


# ============================================================================
# Bridge CRUD
# ============================================================================


class TestReminderCRUD:
    def test_create_reminder_returns_fields(self):
        result = create_reminder("Test reminder", "1h")
        assert "reminder_id" in result
        assert result["text"] == "Test reminder"
        assert "remind_at" in result
        assert "remind_at_local" in result
        assert result["duration"] == "1h"

    def test_create_reminder_stores_memory(self):
        result = create_reminder("Check deployment", "2h")
        reminders = list_reminders()
        assert len(reminders) == 1
        assert reminders[0]["text"] == "Check deployment"
        assert reminders[0]["status"] == "pending"

    def test_list_reminders_sorted(self):
        """Overdue reminders should sort before future ones."""
        # Create a reminder that's already due (very short duration)
        # We'll manipulate the remind_at directly via the store
        from omega.bridge import _get_store

        db = _get_store()

        now = datetime.now(timezone.utc)
        # Overdue reminder
        meta1 = {
            "event_type": "reminder",
            "reminder_status": "pending",
            "remind_at": (now - timedelta(hours=1)).isoformat(),
        }
        db.store(content="Overdue task", metadata=meta1, ttl_seconds=None)

        # Future reminder
        meta2 = {
            "event_type": "reminder",
            "reminder_status": "pending",
            "remind_at": (now + timedelta(hours=1)).isoformat(),
        }
        db.store(content="Future task", metadata=meta2, ttl_seconds=None)

        reminders = list_reminders()
        assert len(reminders) == 2
        assert reminders[0]["text"] == "Overdue task"
        assert reminders[0]["is_overdue"] is True
        assert reminders[1]["text"] == "Future task"
        assert reminders[1]["is_overdue"] is False

    def test_dismiss_reminder(self):
        result = create_reminder("Dismiss me", "1h")
        rid = result["reminder_id"]

        dismiss_result = dismiss_reminder(rid)
        assert dismiss_result["success"] is True

        # Should not appear in default list
        reminders = list_reminders()
        assert len(reminders) == 0

        # Should appear when explicitly including dismissed
        reminders = list_reminders(include_dismissed=True)
        assert len(reminders) == 1
        assert reminders[0]["status"] == "dismissed"

    def test_dismiss_nonexistent(self):
        result = dismiss_reminder("mem-nonexistent1")
        assert result["success"] is False

    def test_dismiss_non_reminder(self):
        """Dismissing a non-reminder memory should fail."""
        from omega.bridge import auto_capture

        auto_capture("Some lesson", "lesson_learned")

        from omega.bridge import _get_store

        db = _get_store()
        # Find the lesson
        with db._lock:
            row = db._conn.execute(
                "SELECT node_id FROM memories WHERE event_type = 'lesson_learned' LIMIT 1"
            ).fetchone()
        assert row is not None
        result = dismiss_reminder(row[0])
        assert result["success"] is False
        assert "not a reminder" in result["error"]

    def test_get_due_reminders_mark_fired(self):
        """mark_fired should transition pending → fired."""
        from omega.bridge import _get_store

        db = _get_store()
        now = datetime.now(timezone.utc)
        meta = {
            "event_type": "reminder",
            "reminder_status": "pending",
            "remind_at": (now - timedelta(minutes=5)).isoformat(),
        }
        db.store(content="Fire me", metadata=meta, ttl_seconds=None)

        due = get_due_reminders(mark_fired=True)
        assert len(due) == 1
        assert due[0]["status"] == "fired"

        # Second call should return nothing (status now "fired", not "pending")
        due2 = get_due_reminders(mark_fired=False)
        assert len(due2) == 0

    def test_same_text_different_times(self):
        """Same text with different durations should create two separate reminders."""
        create_reminder("Check deployment", "1h")
        create_reminder("Check deployment", "2h")

        reminders = list_reminders()
        assert len(reminders) == 2

    def test_create_with_context(self):
        result = create_reminder("Review PR", "30m", context="PR #42 on omega repo")
        reminders = list_reminders()
        assert len(reminders) == 1
        assert reminders[0]["context"] == "PR #42 on omega repo"


# ============================================================================
# MCP Handler Integration
# ============================================================================


class TestReminderHandlers:
    @pytest.mark.asyncio
    async def test_remind_missing_text(self):
        result = await HANDLERS["omega_remind"]({"duration": "1h"})
        assert result.get("isError")

    @pytest.mark.asyncio
    async def test_remind_missing_duration(self):
        result = await HANDLERS["omega_remind"]({"text": "test"})
        assert result.get("isError")

    @pytest.mark.asyncio
    async def test_remind_list_empty(self):
        result = await HANDLERS["omega_remind_list"]({})
        assert not result.get("isError")
        text = result["content"][0]["text"]
        assert "No reminders found" in text

    @pytest.mark.asyncio
    async def test_remind_roundtrip(self):
        """Create → list → dismiss → list (gone)."""
        # Create
        result = await HANDLERS["omega_remind"](
            {"text": "Deploy staging", "duration": "1h", "context": "v2.0 release"}
        )
        assert not result.get("isError")
        text = result["content"][0]["text"]
        assert "Reminder set" in text

        # Extract ID from response
        for line in text.split("\n"):
            if line.startswith("ID:"):
                rid = line.split("ID:")[1].strip()
                break

        # List — should have 1
        result = await HANDLERS["omega_remind_list"]({})
        text = result["content"][0]["text"]
        assert "Deploy staging" in text
        assert "1 found" in text

        # Dismiss
        result = await HANDLERS["omega_remind_dismiss"]({"reminder_id": rid})
        assert not result.get("isError")

        # List — should be empty
        result = await HANDLERS["omega_remind_list"]({})
        text = result["content"][0]["text"]
        assert "No reminders found" in text


# ============================================================================
# Hook Integration
# ============================================================================


class TestReminderHooks:
    def test_session_start_with_due_reminder(self):
        """Session start should include [REMINDER] block when reminders are due."""
        from omega.bridge import _get_store

        db = _get_store()
        now = datetime.now(timezone.utc)
        meta = {
            "event_type": "reminder",
            "reminder_status": "pending",
            "remind_at": (now - timedelta(minutes=10)).isoformat(),
        }
        db.store(content="Urgent: review the plan", metadata=meta, ttl_seconds=None)

        from omega.server.hook_server import handle_session_start

        result = handle_session_start({"session_id": "test-session", "project": ""})
        output = result.get("output", "")
        assert "[REMINDER]" in output
        assert "Urgent: review the plan" in output

    def test_session_start_no_reminders(self):
        """Session start should NOT include [REMINDER] block when no reminders are due."""
        from omega.server.hook_server import handle_session_start

        result = handle_session_start({"session_id": "test-session", "project": ""})
        output = result.get("output", "")
        assert "[REMINDER]" not in output

    def test_surface_memories_reminder_check(self):
        """Surface memories should include due reminders (debounced)."""
        from omega.bridge import _get_store
        from omega.server import hook_server

        # Reset debounce
        hook_server._last_reminder_check = 0.0

        db = _get_store()
        now = datetime.now(timezone.utc)
        meta = {
            "event_type": "reminder",
            "reminder_status": "pending",
            "remind_at": (now - timedelta(minutes=1)).isoformat(),
        }
        db.store(content="Check the build", metadata=meta, ttl_seconds=None)

        result = hook_server.handle_surface_memories({
            "tool_name": "Read",
            "tool_input": '{"file_path": "/tmp/test.py"}',
            "tool_output": "",
            "session_id": "test",
            "project": "",
        })
        output = result.get("output", "")
        assert "[REMINDER]" in output
        assert "Check the build" in output
