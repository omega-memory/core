"""
OMEGA Types - Constants and event type definitions.
"""

from typing import Dict, Optional


# ============================================================================
# TTL Category Constants for Autonomous Memory Capture
# ============================================================================


class TTLCategory:
    """
    Standardized TTL categories for autonomous memory capture.

    Usage:
        from omega.types import TTLCategory
        store.store(content, ttl_seconds=TTLCategory.SHORT_TERM)
    """

    EPHEMERAL = 3600  # 1 hour - temporary context, scratch data
    SHORT_TERM = 86400  # 1 day - blocked context, daily work
    LONG_TERM = 1209600  # 2 weeks - summaries, task completions, decisions
    PERMANENT = None  # Never expires - lessons, preferences, error patterns

    @classmethod
    def for_event_type(cls, event_type: str) -> Optional[int]:
        """Get the appropriate TTL for an event type."""
        return EVENT_TYPE_TTL.get(event_type, cls.LONG_TERM)


class AutoCaptureEventType:
    """Standardized event types for autonomous memory capture."""

    # Core events
    SESSION_SUMMARY = "session_summary"
    TASK_COMPLETION = "task_completion"
    ERROR_PATTERN = "error_pattern"
    LESSON_LEARNED = "lesson_learned"
    DECISION = "decision"
    BLOCKED_CONTEXT = "blocked_context"
    USER_PREFERENCE = "user_preference"
    ADVISOR_INSIGHT = "advisor_insight"

    # Git events
    GIT_COMMIT = "git_commit"
    GIT_MERGE = "git_merge"
    GIT_CONFLICT = "git_conflict"

    # Lifecycle events
    SESSION_START = "session_start"
    SESSION_END = "session_end"
    CONTEXT_WARNING = "context_warning"
    BUDGET_ALERT = "budget_alert"

    # Coordination events
    COORDINATION_SNAPSHOT = "coordination_snapshot"

    # Context virtualization
    CHECKPOINT = "checkpoint"

    # Proactive reminders
    REMINDER = "reminder"


# Map event types to TTL categories
EVENT_TYPE_TTL: Dict[str, Optional[int]] = {
    AutoCaptureEventType.SESSION_SUMMARY: TTLCategory.EPHEMERAL,  # 1 hour: available during session, don't accumulate
    AutoCaptureEventType.TASK_COMPLETION: TTLCategory.LONG_TERM,
    AutoCaptureEventType.ERROR_PATTERN: TTLCategory.PERMANENT,
    AutoCaptureEventType.LESSON_LEARNED: TTLCategory.PERMANENT,
    AutoCaptureEventType.DECISION: TTLCategory.LONG_TERM,
    AutoCaptureEventType.BLOCKED_CONTEXT: TTLCategory.SHORT_TERM,
    AutoCaptureEventType.USER_PREFERENCE: TTLCategory.PERMANENT,
    AutoCaptureEventType.ADVISOR_INSIGHT: TTLCategory.LONG_TERM,
    AutoCaptureEventType.GIT_COMMIT: TTLCategory.LONG_TERM,
    AutoCaptureEventType.GIT_MERGE: TTLCategory.LONG_TERM,
    AutoCaptureEventType.GIT_CONFLICT: TTLCategory.PERMANENT,
    AutoCaptureEventType.SESSION_START: TTLCategory.SHORT_TERM,
    AutoCaptureEventType.SESSION_END: TTLCategory.LONG_TERM,
    AutoCaptureEventType.CONTEXT_WARNING: TTLCategory.SHORT_TERM,
    AutoCaptureEventType.BUDGET_ALERT: TTLCategory.LONG_TERM,
    AutoCaptureEventType.COORDINATION_SNAPSHOT: TTLCategory.SHORT_TERM,
    # User-facing types (from legacy/migration)
    "user_fact": TTLCategory.PERMANENT,  # Facts about the user (similar to user_preference)
    "user_prompt": TTLCategory.LONG_TERM,  # Captured user prompts
    "system_event": TTLCategory.SHORT_TERM,  # System-level events
    # Research & evaluation (permanent)
    "sota_research": TTLCategory.PERMANENT,
    "research_report": TTLCategory.PERMANENT,
    "preference_generated": TTLCategory.PERMANENT,
    # Long-term (2 weeks)
    "reflexion": TTLCategory.LONG_TERM,
    "outcome_evaluation": TTLCategory.LONG_TERM,
    "self_reflection": TTLCategory.LONG_TERM,
    "advisor_action_outcome": TTLCategory.LONG_TERM,
    "benchmark_update": TTLCategory.LONG_TERM,
    "file_conflict": TTLCategory.LONG_TERM,
    "session_respawn": TTLCategory.LONG_TERM,
    "memory": TTLCategory.SHORT_TERM,  # Generic type with no query path; expire quickly
    # Context virtualization (7 days)
    AutoCaptureEventType.CHECKPOINT: 604800,  # 7 days
    # Proactive reminders (permanent until dismissed)
    AutoCaptureEventType.REMINDER: None,
    # Short-term (1 day)
    "sota_scan": TTLCategory.SHORT_TERM,
    "merge_claim": TTLCategory.SHORT_TERM,
    "merge_release": TTLCategory.SHORT_TERM,
    "file_claimed": TTLCategory.SHORT_TERM,
    "file_released": TTLCategory.SHORT_TERM,
    "branch_claimed": TTLCategory.SHORT_TERM,
    "branch_released": TTLCategory.SHORT_TERM,
    "test": TTLCategory.SHORT_TERM,
    "file_summary": TTLCategory.SHORT_TERM,
    # Ephemeral (1 hour)
    "code_chunk": TTLCategory.EPHEMERAL,
}


__all__ = [
    "TTLCategory",
    "AutoCaptureEventType",
    "EVENT_TYPE_TTL",
]
