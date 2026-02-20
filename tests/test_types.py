"""Tests for omega.types — TTL constants, event types, and TTL mapping."""
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from omega.types import TTLCategory, AutoCaptureEventType, EVENT_TYPE_TTL


class TestAutoCaptureEventTypesHaveTTL:
    def test_all_class_event_types_in_ttl_map(self):
        """Every AutoCaptureEventType constant should have a TTL mapping."""
        event_types = [
            v for k, v in vars(AutoCaptureEventType).items()
            if not k.startswith("_") and isinstance(v, str)
        ]
        missing = [et for et in event_types if et not in EVENT_TYPE_TTL]
        assert missing == [], f"Event types missing TTL mapping: {missing}"


class TestForEventType:
    def test_known_type_returns_correct_ttl(self):
        assert TTLCategory.for_event_type("lesson_learned") == TTLCategory.PERMANENT

    def test_session_summary_is_ephemeral(self):
        assert TTLCategory.for_event_type("session_summary") == TTLCategory.EPHEMERAL

    def test_unknown_type_defaults_to_long_term(self):
        assert TTLCategory.for_event_type("completely_unknown_type") == TTLCategory.LONG_TERM


class TestAllExports:
    def test_all_contains_expected(self):
        import omega.types
        assert set(omega.types.__all__) == {"TTLCategory", "AutoCaptureEventType", "EVENT_TYPE_TTL"}
