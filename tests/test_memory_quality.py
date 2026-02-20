"""Tests for memory quality gates (Phase 2-3 of noise reduction).

Tests:
- Content blocklist rejects system noise
- Min-length gate rejects short content
- Dedup thresholds catch more duplicates
- Error burst detection prevents flood
- Confidence metadata is set correctly
- Session summary excluded from user-facing queries
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture
def bridge_env(tmp_omega_dir):
    """Set up bridge with a fresh store."""
    from omega.bridge import reset_memory
    reset_memory()
    yield tmp_omega_dir
    reset_memory()


class TestContentBlocklist:
    """Phase 2A: Content blocklist rejects system noise."""

    def test_broadcast_blocked(self, bridge_env):
        from omega.bridge import auto_capture
        result = auto_capture(
            content="[BROADCAST from agent-1] starting work on bridge.py",
            event_type="decision",
        )
        assert "Blocked" in result
        assert "system noise" in result

    def test_work_breadcrumb_blocked(self, bridge_env):
        from omega.bridge import auto_capture
        result = auto_capture(
            content="[WORK BREADCRUMB] editing file src/omega/bridge.py at line 42",
            event_type="decision",
        )
        assert "Blocked" in result

    def test_work_dispatch_blocked(self, bridge_env):
        from omega.bridge import auto_capture
        result = auto_capture(
            content="[WORK DISPATCH] dispatching task to agent-2 for bridge.py refactor",
            event_type="decision",
        )
        assert "Blocked" in result

    def test_task_notification_blocked(self, bridge_env):
        from omega.bridge import auto_capture
        result = auto_capture(
            content="<task-notification><task-id>abc123</task-id><status>completed</status></task-notification>",
            event_type="decision",
        )
        assert "Blocked" in result

    def test_decision_task_notification_blocked(self, bridge_env):
        from omega.bridge import auto_capture
        result = auto_capture(
            content="Decision: <task-notification> completed task abc for agent testing",
            event_type="decision",
        )
        assert "Blocked" in result

    def test_normal_content_passes(self, bridge_env):
        from omega.bridge import auto_capture
        result = auto_capture(
            content="Decided to use SQLite for the memory backend because it reduces RAM from 372MB to 31MB",
            event_type="decision",
        )
        assert "Blocked" not in result
        assert "Captured" in result or "Evolved" in result or "Deduped" in result


class TestMinLengthGate:
    """Phase 2A: Min-length gate rejects short hook-sourced content."""

    def test_short_hook_content_blocked(self, bridge_env):
        from omega.bridge import auto_capture
        result = auto_capture(
            content="test",
            event_type="decision",
            metadata={"source": "auto_capture_hook"},
        )
        assert "Blocked" in result
        assert "too short" in result

    def test_short_hook_content_39_chars_blocked(self, bridge_env):
        from omega.bridge import auto_capture
        result = auto_capture(
            content="a" * 39,
            event_type="decision",
            metadata={"source": "auto_plan_capture"},
        )
        assert "Blocked" in result

    def test_min_length_hook_content_passes(self, bridge_env):
        from omega.bridge import auto_capture
        result = auto_capture(
            content="a" * 40 + " this is a real decision about architecture",
            event_type="decision",
            metadata={"source": "auto_capture_hook"},
        )
        assert "Blocked" not in result

    def test_short_direct_api_allowed(self, bridge_env):
        """Direct API calls (no hook source) should not be blocked by min-length."""
        from omega.bridge import auto_capture
        result = auto_capture(
            content="Short but valid decision",
            event_type="decision",
        )
        assert "too short" not in result

    def test_user_preference_short_allowed(self, bridge_env):
        from omega.bridge import auto_capture
        result = auto_capture(
            content="prefers dark mode",
            event_type="user_preference",
            metadata={"source": "auto_capture_hook"},
        )
        # user_preference should not be blocked by min-length
        assert "too short" not in result


class TestDedupThresholds:
    """Phase 2B: Lowered thresholds catch more duplicates."""

    def test_similar_decisions_deduped(self, bridge_env):
        from omega.bridge import auto_capture
        # First capture — long content with many words for high Jaccard overlap
        r1 = auto_capture(
            content="Decided to use PostgreSQL with connection pooling via pgbouncer for the database layer in the production environment for improved performance and reliability",
            event_type="decision",
        )
        assert "Captured" in r1

        # Very similar content — only 1 word different (Jaccard ~0.88)
        r2 = auto_capture(
            content="Decided to use PostgreSQL with connection pooling via pgbouncer for the database layer in the staging environment for improved performance and reliability",
            event_type="decision",
        )
        assert "Deduped" in r2

    def test_different_decisions_not_deduped(self, bridge_env):
        from omega.bridge import auto_capture
        r1 = auto_capture(
            content="Decided to use PostgreSQL with connection pooling for the database layer in production environment",
            event_type="decision",
        )
        assert "Captured" in r1

        r2 = auto_capture(
            content="Decided to switch from REST API to GraphQL for the frontend communication layer entirely",
            event_type="decision",
        )
        assert "Captured" in r2 or "Evolved" in r2


class TestErrorBurstDetection:
    """Phase 3B: Error burst detection prevents test-run floods."""

    def test_first_error_captured(self, bridge_env):
        from omega.bridge import auto_capture
        result = auto_capture(
            content="Error encountered: ModuleNotFoundError: No module named 'nonexistent_package' in test suite",
            event_type="error_pattern",
            session_id="test-session-burst",
        )
        assert "Blocked" not in result

    def test_repeated_similar_errors_blocked(self, bridge_env):
        from omega.bridge import auto_capture
        sid = "test-session-burst-2"

        # Store 3 similar errors to trigger burst detection
        for i in range(3):
            auto_capture(
                content=f"Error encountered: TypeError: cannot read property 'length' of undefined in component {i}",
                event_type="error_pattern",
                session_id=sid,
            )

        # 4th similar error should be blocked
        result = auto_capture(
            content="Error encountered: TypeError: cannot read property 'length' of undefined in component 99",
            event_type="error_pattern",
            session_id=sid,
        )
        # Either deduped or burst-blocked
        assert "Deduped" in result or "Blocked" in result


class TestConfidenceMetadata:
    """Phase 3C: Confidence metadata is set correctly."""

    def test_user_remember_high_confidence(self, bridge_env):
        from omega.bridge import remember
        from omega.bridge import _get_store
        remember("Always use type hints in Python code")
        store = _get_store()
        # Find the memory we just stored
        memories = store.get_by_type("user_preference", limit=1)
        assert len(memories) >= 1
        assert memories[0].metadata.get("capture_confidence") == "high"

    def test_auto_capture_hook_medium_confidence(self, bridge_env):
        """Auto-captured content from hooks gets medium confidence."""
        from omega.bridge import auto_capture, _get_store
        auto_capture(
            content="Decided to refactor the authentication module to use JWT tokens instead of session cookies",
            event_type="decision",
            metadata={"source": "auto_capture_hook"},
        )
        store = _get_store()
        memories = store.get_by_type("decision", limit=1)
        assert len(memories) >= 1
        assert memories[0].metadata.get("capture_confidence") == "medium"

    def test_direct_api_high_confidence(self, bridge_env):
        """Direct API calls (no source) get high confidence."""
        from omega.bridge import auto_capture, _get_store
        auto_capture(
            content="Decided to use SQLite instead of PostgreSQL for the persistence layer of the memory system",
            event_type="decision",
        )
        store = _get_store()
        memories = store.get_by_type("decision", limit=1)
        assert len(memories) >= 1
        assert memories[0].metadata.get("capture_confidence") == "high"


class TestSessionSummaryInfrastructure:
    """Phase 2G: session_summary excluded from user-facing queries."""

    def test_session_summary_in_infrastructure_types(self):
        from omega.sqlite_store import SQLiteStore
        assert "session_summary" in SQLiteStore._INFRASTRUCTURE_TYPES

    def test_session_summary_hidden_from_query(self, bridge_env):
        from omega.bridge import auto_capture, _get_store
        auto_capture(
            content="Session summary: worked on memory quality improvements and noise reduction across multiple files",
            event_type="session_summary",
            session_id="test-session",
        )
        store = _get_store()
        # Query without include_infrastructure should not return session_summary
        results = store.query("memory quality improvements", limit=10)
        summary_results = [
            r for r in results
            if (r.metadata or {}).get("event_type") == "session_summary"
        ]
        assert len(summary_results) == 0

    def test_session_summary_ttl_ephemeral(self):
        from omega.types import TTLCategory, AutoCaptureEventType, EVENT_TYPE_TTL
        assert EVENT_TYPE_TTL[AutoCaptureEventType.SESSION_SUMMARY] == TTLCategory.EPHEMERAL


class TestBlocklistScoping:
    """Contains-blocklist should only apply to hook-sourced content."""

    def test_direct_api_with_error_key_not_blocked(self, bridge_env):
        """Direct omega_store with content containing 'error:' should not be blocked."""
        from omega.bridge import auto_capture
        result = auto_capture(
            content='Always check "error": key in JSON responses before assuming success',
            event_type="lesson_learned",
        )
        assert "Blocked" not in result
        assert "Captured" in result or "Evolved" in result

    def test_direct_api_with_stderr_not_blocked(self, bridge_env):
        """Direct API call mentioning stderr should pass."""
        from omega.bridge import auto_capture
        result = auto_capture(
            content='The "stderr": field contains error output from subprocess.run calls and should be logged',
            event_type="lesson_learned",
        )
        assert "Blocked" not in result

    def test_hook_with_error_key_still_blocked(self, bridge_env):
        """Hook-sourced content with JSON noise patterns should still be blocked."""
        from omega.bridge import auto_capture
        result = auto_capture(
            content='{"error": "command not found", "stderr": "bash: foo: not found"}',
            event_type="error_pattern",
            metadata={"source": "auto_capture_hook"},
        )
        assert "Blocked" in result
        assert "system noise" in result

    def test_startswith_blocklist_still_applies_to_all(self, bridge_env):
        """Startswith patterns should block regardless of source."""
        from omega.bridge import auto_capture
        result = auto_capture(
            content="[BROADCAST from agent] important decision about architecture",
            event_type="decision",
        )
        assert "Blocked" in result


class TestBridgeExports:
    """__all__ should include all public functions."""

    def test_all_exports_complete(self):
        import omega.bridge as b
        # These 4 were previously missing from __all__
        assert "check_constraints" in b.__all__
        assert "list_constraints" in b.__all__
        assert "save_constraints" in b.__all__
        assert "get_cross_project_lessons" in b.__all__

    def test_no_duplicate_tag_tools(self):
        from omega.bridge import _TAG_TOOLS
        # _TAG_TOOLS is a set, but verify no conceptual duplicates
        assert isinstance(_TAG_TOOLS, set)
