"""
OMEGA Preference Extractor - Phase 1 of Memory Advancement

Automatically detects and stores user preferences from conversation
without requiring explicit "remember" commands.

Includes:
- Pattern-based implicit preference extraction (Phase 1A)
- Preference conflict/contradiction detection (Phase 2A)
- MCP tool interface for agent-driven extraction (Phase 1C)
"""

import re
from typing import Any, Dict, List, Optional, Tuple

import logging

logger = logging.getLogger("omega.preferences")


class PreferenceExtractor:
    """Extract implicit preferences from user messages."""

    # (pattern, confidence) - higher confidence = more explicit preference signal
    PATTERNS: List[Tuple[str, float]] = [
        # Explicit preference signals
        (r"(?:I |i )(?:prefer|always|never|like to|want to|don't like)\b(.+)", 0.7),
        (r"(?:please |)(?:always|never)\b(.+)", 0.6),
        (r"(?:use |don't use )\b(.+?)(?:\s+(?:for|when|in)\b|$)", 0.6),
        # Style preferences
        (r"(?:I |i )(?:like|love|hate|dislike)\s+(.+?)(?:\.|$)", 0.6),
        (r"(?:I |i )(?:don't want|don't need|don't like)\s+(.+?)(?:\.|$)", 0.65),
        # Instruction patterns
        (r"(?:make sure|ensure|be sure)\s+(?:to\s+)?(.+?)(?:\.|$)", 0.5),
        (r"(?:from now on|going forward|in the future)\s*[,]?\s*(.+?)(?:\.|$)", 0.7),
    ]

    # Negation words for contradiction detection
    NEGATION_WORDS = {"don't", "dont", "don", "never", "no", "not", "without", "avoid", "stop"}
    POSITIVE_WORDS = {"always", "prefer", "use", "like", "want", "with", "include"}

    def extract(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract preference-like statements from text.

        Returns:
            List of {content, confidence, pattern_matched} dicts
        """
        results = []
        seen_contents = set()

        for pattern, confidence in self.PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                content = match.group(0).strip().rstrip(".,;:")

                if len(content) < 10 or len(content) > 300:
                    continue

                content_lower = content.lower()
                if content_lower in seen_contents:
                    continue
                seen_contents.add(content_lower)

                results.append(
                    {
                        "content": content,
                        "confidence": confidence,
                        "pattern_matched": pattern[:40],
                    }
                )

        return results

    def detect_contradiction(self, new_pref: str, existing_prefs: List[Any]) -> List[Dict[str, Any]]:
        """
        Detect if a new preference contradicts existing ones.

        Uses topic word overlap + negation polarity flip detection.
        """
        contradictions = []
        new_words = set(re.findall(r"\b\w{3,}\b", new_pref.lower()))
        new_has_negation = bool(new_words & self.NEGATION_WORDS)
        new_topic_words = new_words - self.NEGATION_WORDS - self.POSITIVE_WORDS

        for existing in existing_prefs:
            existing_content = existing.content if hasattr(existing, "content") else str(existing)
            existing_id = existing.id if hasattr(existing, "id") else ""

            existing_words = set(re.findall(r"\b\w{3,}\b", existing_content.lower()))
            existing_has_negation = bool(existing_words & self.NEGATION_WORDS)
            existing_topic_words = existing_words - self.NEGATION_WORDS - self.POSITIVE_WORDS

            if not new_topic_words or not existing_topic_words:
                continue

            intersection = new_topic_words & existing_topic_words
            union = new_topic_words | existing_topic_words
            jaccard = len(intersection) / len(union) if union else 0

            if jaccard < 0.3:
                continue

            polarity_flip = new_has_negation != existing_has_negation

            if polarity_flip:
                contradictions.append(
                    {
                        "existing_id": existing_id,
                        "existing_content": existing_content[:200],
                        "reason": f"Polarity flip detected (topic overlap: {jaccard:.0%})",
                        "jaccard": jaccard,
                    }
                )

        return contradictions


# Module-level singleton
_extractor = PreferenceExtractor()


def extract_preferences(text: str) -> List[Dict[str, Any]]:
    """Module-level convenience function."""
    return _extractor.extract(text)


def detect_contradictions(new_pref: str, existing_prefs: List[Any]) -> List[Dict[str, Any]]:
    """Module-level convenience function."""
    return _extractor.detect_contradiction(new_pref, existing_prefs)


def store_extracted_preferences(
    text: str,
    session_id: Optional[str] = None,
    auto_resolve_conflicts: bool = True,
) -> Dict[str, Any]:
    """
    Full pipeline: extract preferences from text, check for conflicts,
    store new ones, and mark superseded ones.
    """
    extracted = extract_preferences(text)
    if not extracted:
        return {"extracted": 0, "stored": 0, "conflicts": 0, "superseded": 0}

    try:
        from omega.bridge import _get_store

        memory = _get_store()
    except ImportError:
        logger.warning("OMEGA not available, cannot store preferences")
        return {
            "extracted": len(extracted),
            "stored": 0,
            "conflicts": 0,
            "superseded": 0,
            "error": "OMEGA not available",
        }

    stored_count = 0
    conflict_count = 0
    superseded_count = 0

    for pref in extracted:
        content = pref["content"]
        confidence = pref["confidence"]

        # Check for conflicts with existing preferences
        existing_prefs = memory.get_by_type("user_preference", limit=20)
        contradictions = detect_contradictions(content, existing_prefs)

        if contradictions:
            conflict_count += len(contradictions)

            if auto_resolve_conflicts:
                for contradiction in contradictions:
                    old_id = contradiction["existing_id"]
                    if old_id:
                        memory.record_feedback(old_id, "outdated", reason=f"Superseded by: {content[:80]}")
                        superseded_count += 1
                        logger.info(f"Superseded preference {old_id[:12]}: {contradiction['reason']}")

        # Store the new preference
        memory.store(
            content=content,
            session_id=session_id,
            metadata={
                "event_type": "user_preference",
                "source": "implicit_extraction",
                "confidence": confidence,
                "pattern": pref.get("pattern_matched", ""),
                "conflicts_resolved": len(contradictions) if auto_resolve_conflicts else 0,
            },
            ttl_seconds=None,  # Permanent
        )
        stored_count += 1
        logger.info(f"Stored implicit preference (confidence={confidence}): {content[:60]}")

    return {
        "extracted": len(extracted),
        "stored": stored_count,
        "conflicts": conflict_count,
        "superseded": superseded_count,
    }
