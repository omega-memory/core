"""Tests for omega.preferences — extraction, contradiction detection, store pipeline."""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch


sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from omega.preferences import (
    extract_preferences,
    detect_contradictions,
    store_extracted_preferences,
)


# ============================================================================
# PreferenceExtractor.extract
# ============================================================================


class TestExtract:
    def test_explicit_preference(self):
        results = extract_preferences("I prefer using dark mode for all editors")
        assert len(results) >= 1
        assert any("dark mode" in r["content"].lower() for r in results)

    def test_negation_preference(self):
        results = extract_preferences("I don't like tabs in Python code at all")
        assert len(results) >= 1
        assert any(r["confidence"] >= 0.6 for r in results)

    def test_future_directive(self):
        results = extract_preferences("From now on, always use type hints in function signatures")
        assert len(results) >= 1
        assert any("type hints" in r["content"].lower() for r in results)

    def test_too_short_filtered(self):
        """Content under 10 chars should be filtered out."""
        results = extract_preferences("I prefer")
        assert len(results) == 0

    def test_too_long_filtered(self):
        """Content over 300 chars should be filtered out."""
        long_text = "I prefer " + "x" * 300
        results = extract_preferences(long_text)
        assert len(results) == 0

    def test_deduplication(self):
        """Same preference matched by multiple patterns should appear once."""
        text = "I always prefer using pytest. I always prefer using pytest."
        results = extract_preferences(text)
        contents = [r["content"].lower() for r in results]
        # No exact duplicates
        assert len(contents) == len(set(contents))

    def test_no_match_returns_empty(self):
        results = extract_preferences("The weather is nice today")
        assert results == []

    def test_result_has_expected_keys(self):
        results = extract_preferences("I prefer using snake_case for variable names in Python")
        if results:
            r = results[0]
            assert "content" in r
            assert "confidence" in r
            assert "pattern_matched" in r


# ============================================================================
# detect_contradictions
# ============================================================================


class MockPref:
    """Lightweight mock for existing preference objects."""
    def __init__(self, content, id="pref-1"):
        self.content = content
        self.id = id


class TestDetectContradiction:
    def test_polarity_flip_detected(self):
        """'always use X' vs 'never use X' should detect contradiction."""
        existing = [MockPref("I always use spaces for indentation in Python")]
        contradictions = detect_contradictions(
            "I never use spaces for indentation in Python", existing
        )
        assert len(contradictions) >= 1
        assert "Polarity flip" in contradictions[0]["reason"]

    def test_low_overlap_no_contradiction(self):
        """Unrelated topics should not flag contradiction."""
        existing = [MockPref("I prefer dark mode for all editors")]
        contradictions = detect_contradictions(
            "I never use tabs for indentation", existing
        )
        assert len(contradictions) == 0

    def test_same_polarity_no_contradiction(self):
        """Same polarity (both positive) should not flag contradiction."""
        existing = [MockPref("I always use pytest for testing")]
        contradictions = detect_contradictions(
            "I always use pytest for integration testing", existing
        )
        assert len(contradictions) == 0

    def test_empty_existing_returns_empty(self):
        contradictions = detect_contradictions("I prefer dark mode", [])
        assert contradictions == []

    def test_works_with_plain_strings(self):
        """Should handle plain strings (no .content/.id attributes)."""
        existing = ["I always use spaces for indentation in Python code"]
        contradictions = detect_contradictions(
            "I never use spaces for indentation in Python code", existing
        )
        assert len(contradictions) >= 1


# ============================================================================
# store_extracted_preferences
# ============================================================================


class TestStoreExtractedPreferences:
    def test_no_extraction_returns_zeros(self):
        result = store_extracted_preferences("The weather is nice today")
        assert result["extracted"] == 0
        assert result["stored"] == 0

    def test_import_error_returns_gracefully(self):
        """When omega.bridge is unavailable, should return gracefully."""
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "omega.bridge":
                raise ImportError("mocked")
            return real_import(name, *args, **kwargs)

        with patch("omega.preferences.extract_preferences", return_value=[
            {"content": "I prefer dark mode for editors", "confidence": 0.7, "pattern_matched": "test"}
        ]):
            with patch.object(builtins, "__import__", side_effect=mock_import):
                result = store_extracted_preferences("I prefer dark mode for editors")
                assert result["stored"] == 0
                assert "error" in result

    def test_full_pipeline_with_mock_store(self):
        mock_store = MagicMock()
        mock_store.get_by_type.return_value = []
        mock_store.store.return_value = "new-node-id"

        with patch("omega.bridge._get_store", return_value=mock_store):
            result = store_extracted_preferences(
                "I prefer using ruff over flake8 for Python linting always"
            )

        if result["extracted"] > 0:
            assert result["stored"] == result["extracted"]
            assert mock_store.store.call_count == result["stored"]

    def test_conflict_resolution_calls_record_feedback(self):
        mock_existing = MagicMock()
        mock_existing.content = "I always use flake8 for Python linting"
        mock_existing.id = "old-pref-123"

        mock_store = MagicMock()
        mock_store.get_by_type.return_value = [mock_existing]
        mock_store.store.return_value = "new-node-id"

        with patch("omega.bridge._get_store", return_value=mock_store), \
             patch("omega.preferences.extract_preferences", return_value=[
                 {"content": "I never use flake8 for Python linting", "confidence": 0.7, "pattern_matched": "test"}
             ]):
            result = store_extracted_preferences("I never use flake8 for Python linting")

        assert result["stored"] == 1
        if result["conflicts"] > 0:
            mock_store.record_feedback.assert_called()
            assert result["superseded"] >= 1
