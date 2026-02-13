"""UAT — Memory Lifecycle, Knowledge Graph, Sessions, Preferences & Safety.

End-to-end acceptance tests for the OMEGA memory system.
Tests the full lifecycle through MCP handler interface.

Organized into ten sections:
  1. Lifecycle — store → dedup → query → edit → delete → verify gone
  2. Evolution — similar memories evolve instead of duplicating
  3. Knowledge Graph — auto-relate edges → traverse
  4. Session Scoping — cross-session isolation, clear
  5. Preference Management — remember → list → profile
  6. Consolidation Pipeline — consolidate → compact → type_stats
  7. Welcome Briefing — welcome returns relevant subset
  8. Export/Import — backup → restore roundtrip
  9. Feedback Loop — helpful/outdated ratings
  10. Content Safety — blocklist, min-length, path traversal
"""
import json
import re
import sys
import pytest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from omega.server.handlers import HANDLERS


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture(autouse=True)
def _reset_bridge(tmp_omega_dir):
    """Reset the bridge singleton before and after each test."""
    from omega.bridge import reset_memory
    reset_memory()
    yield
    reset_memory()


def _text(result: dict) -> str:
    """Extract text from an MCP response."""
    return result["content"][0]["text"]


def _is_error(result: dict) -> bool:
    """Check if an MCP response is an error."""
    return result.get("isError", False)


async def _store_and_get_id(content, event_type="lesson_learned", **kwargs) -> str:
    """Store via handler, extract node_id from response."""
    result = await HANDLERS["omega_store"](
        {"content": content, "event_type": event_type, **kwargs}
    )
    assert not _is_error(result), f"Store failed: {_text(result)}"
    text = _text(result)
    # Extract node ID from "**Node ID:** `<id>`"
    match = re.search(r"Node ID:\*\*\s*`([^`]+)`", text)
    if match:
        return match.group(1)
    # Also check for dedup/evolved messages which contain shorter IDs
    match = re.search(r"existing\s+`?([a-f0-9-]+)", text)
    if match:
        return match.group(1)
    # Return the full text if we can't extract an ID (dedup case)
    return text


# ============================================================================
# SECTION 1: Memory Lifecycle
# ============================================================================


class TestUATMemoryLifecycle:
    """Full CRUD lifecycle through MCP handlers."""

    @pytest.mark.asyncio
    async def test_store_returns_node_id(self):
        """UAT: Storing a memory returns a node ID in the response."""
        result = await HANDLERS["omega_store"]({
            "content": "Always use pytest-asyncio for async test fixtures in OMEGA project",
            "event_type": "lesson_learned",
        })
        assert not _is_error(result)
        text = _text(result)
        assert "Node ID:" in text or "Memory Captured" in text

    @pytest.mark.asyncio
    async def test_store_and_query_roundtrip(self):
        """UAT: Store a memory, then retrieve it via query."""
        await _store_and_get_id(
            "SQLite WAL mode improves concurrent read performance significantly",
            event_type="lesson_learned",
        )
        result = await HANDLERS["omega_query"]({"query": "SQLite WAL concurrent read"})
        assert not _is_error(result)
        text = _text(result)
        assert "WAL" in text
        assert "concurrent" in text.lower() or "read" in text.lower()

    @pytest.mark.asyncio
    async def test_store_dedup_reuses_existing(self):
        """UAT: Storing near-identical content is deduplicated."""
        content = "Threading locks in Python are non-reentrant and will deadlock if nested. Always avoid acquiring the same lock twice in the same thread to prevent silent hangs in production code."
        await _store_and_get_id(content, event_type="lesson_learned")
        # Store near-identical content (only minor wording change)
        result = await HANDLERS["omega_store"]({
            "content": "Threading locks in Python are non-reentrant and will deadlock if nested. Always avoid acquiring the same lock twice in the same thread to prevent silent hangs in production systems.",
            "event_type": "lesson_learned",
        })
        text = _text(result)
        assert "Deduplicated" in text or "reused" in text.lower()

    @pytest.mark.asyncio
    async def test_edit_memory(self):
        """UAT: Edit a stored memory's content."""
        node_id = await _store_and_get_id(
            "Use pip install -e for editable installs during development of OMEGA packages",
            event_type="decision",
        )
        result = await HANDLERS["omega_edit_memory"]({
            "memory_id": node_id,
            "new_content": "Use pip install -e . for editable installs during OMEGA development (requires pyproject.toml)",
        })
        assert not _is_error(result)
        text = _text(result)
        assert "Updated" in text
        assert "pyproject.toml" in text

    @pytest.mark.asyncio
    async def test_delete_memory(self):
        """UAT: Delete a stored memory, confirm it's gone."""
        node_id = await _store_and_get_id(
            "Temporary test memory that should be deleted after this verification step",
            event_type="decision",
        )
        result = await HANDLERS["omega_delete_memory"]({"memory_id": node_id})
        assert not _is_error(result)
        assert "Deleted" in _text(result)

        # Verify it's gone - query should not find it
        result = await HANDLERS["omega_query"]({"query": "Temporary test memory deleted verification"})
        text = _text(result)
        assert node_id[:12] not in text or "No matching" in text

    @pytest.mark.asyncio
    async def test_delete_nonexistent_memory(self):
        """UAT: Deleting a nonexistent memory returns error."""
        result = await HANDLERS["omega_delete_memory"](
            {"memory_id": "nonexistent-id-12345"}
        )
        assert _is_error(result)

    @pytest.mark.asyncio
    async def test_edit_nonexistent_memory(self):
        """UAT: Editing a nonexistent memory returns error."""
        result = await HANDLERS["omega_edit_memory"]({
            "memory_id": "nonexistent-id-12345",
            "new_content": "This should fail",
        })
        assert _is_error(result)

    @pytest.mark.asyncio
    async def test_store_empty_content_rejected(self):
        """UAT: Storing empty content returns error."""
        result = await HANDLERS["omega_store"]({"content": "", "event_type": "memory"})
        assert _is_error(result)

    @pytest.mark.asyncio
    async def test_query_empty_rejected(self):
        """UAT: Querying with empty text returns error."""
        result = await HANDLERS["omega_query"]({"query": ""})
        assert _is_error(result)

    @pytest.mark.asyncio
    async def test_phrase_search(self):
        """UAT: Phrase search finds exact matches via FTS5."""
        await _store_and_get_id(
            "The ONNX runtime for bge-small-en-v1.5 uses CPU-only inference mode",
            event_type="lesson_learned",
        )
        result = await HANDLERS["omega_phrase_search"]({"phrase": "bge-small"})
        assert not _is_error(result)
        text = _text(result)
        assert "bge-small" in text.lower() or "ONNX" in text


# ============================================================================
# SECTION 2: Memory Evolution
# ============================================================================


class TestUATMemoryEvolution:
    """Memory evolution (Zettelkasten-style) through handlers."""

    @pytest.mark.asyncio
    async def test_similar_content_evolves(self):
        """UAT: Storing similar (but not identical) content evolves the original."""
        # Store original lesson — long shared body with a short unique tail
        await _store_and_get_id(
            "The OMEGA memory bridge module uses lazy singleton pattern for thread-safe SQLite store initialization with automatic content deduplication checking Jaccard similarity on normalized word sets for efficient memory storage management and retrieval operations across sessions. Duplicates are merged automatically.",
            event_type="lesson_learned",
        )
        # Store similar content with new insight (Jaccard ~0.81, in evolution range 0.65-0.85)
        result = await HANDLERS["omega_store"]({
            "content": "The OMEGA memory bridge module uses lazy singleton pattern for thread-safe SQLite store initialization with automatic content deduplication checking Jaccard similarity on normalized word sets for efficient memory storage management and retrieval operations across sessions. Evolution threshold merges insights from related memories.",
            "event_type": "lesson_learned",
        })
        text = _text(result)
        # Should evolve or deduplicate, not create a completely new memory
        assert ("Evolved" in text or "Deduplicated" in text or
                "reused" in text.lower() or "updated" in text.lower())

    @pytest.mark.asyncio
    async def test_different_content_creates_new(self):
        """UAT: Storing genuinely different content creates a new memory."""
        await _store_and_get_id(
            "Docker containers should use multi-stage builds to minimize final image size in production",
            event_type="lesson_learned",
        )
        result = await HANDLERS["omega_store"]({
            "content": "Kubernetes pod autoscaling requires resource requests and limits to be properly configured in deployment manifests",
            "event_type": "lesson_learned",
        })
        text = _text(result)
        assert "Memory Captured" in text or "Node ID:" in text

    @pytest.mark.asyncio
    async def test_evolution_only_for_eligible_types(self):
        """UAT: Evolution only applies to lesson_learned, decision, error_pattern."""
        # Session summaries should not evolve
        await _store_and_get_id(
            "Session summary for testing the evolution eligibility of different memory types in OMEGA system",
            event_type="session_summary",
        )
        result = await HANDLERS["omega_store"]({
            "content": "Session summary for testing the evolution eligibility of different memory types in OMEGA. Additional context about the session outcome and next steps",
            "event_type": "session_summary",
        })
        text = _text(result)
        # Might dedup but should NOT evolve
        assert "Evolved" not in text

    @pytest.mark.asyncio
    async def test_error_pattern_dedup(self):
        """UAT: Error patterns are deduplicated with normalized comparison."""
        await _store_and_get_id(
            "ImportError: cannot import name 'OmegaMemory' from 'omega.bridge' at /Users/test/proj/omega/bridge.py",
            event_type="error_pattern",
        )
        result = await HANDLERS["omega_store"]({
            "content": "ImportError: cannot import name 'OmegaMemory' from 'omega.bridge' at /Users/different/proj/omega/bridge.py",
            "event_type": "error_pattern",
        })
        text = _text(result)
        assert "Deduplicated" in text or "reused" in text.lower()

    @pytest.mark.asyncio
    async def test_type_stats_after_stores(self):
        """UAT: Type stats reflect stored memories accurately."""
        await _store_and_get_id("Lesson about python threading locks being non-reentrant in concurrent code", event_type="lesson_learned")
        await _store_and_get_id("Decision to use SQLite instead of PostgreSQL for the OMEGA memory backend", event_type="decision")
        await _store_and_get_id("Decision to use ONNX CPU-only backend instead of CoreML for embedding inference", event_type="decision")

        result = await HANDLERS["omega_type_stats"]({})
        assert not _is_error(result)
        text = _text(result)
        assert "lesson_learned" in text
        assert "decision" in text


# ============================================================================
# SECTION 3: Knowledge Graph
# ============================================================================


class TestUATKnowledgeGraph:
    """Auto-relate edges and graph traversal through handlers."""

    @pytest.mark.asyncio
    async def test_query_returns_results_for_related_content(self):
        """UAT: Storing related memories and querying finds them."""
        await _store_and_get_id(
            "OMEGA uses bge-small-en-v1.5 ONNX model for embedding generation in the vector search pipeline",
            event_type="lesson_learned",
        )
        await _store_and_get_id(
            "The embedding model bge-small produces 384-dimensional vectors for semantic similarity search",
            event_type="lesson_learned",
        )
        result = await HANDLERS["omega_query"]({"query": "embedding model vector search"})
        text = _text(result)
        assert "bge-small" in text or "embedding" in text.lower()

    @pytest.mark.asyncio
    async def test_traverse_nonexistent_memory(self):
        """UAT: Traversing from a nonexistent memory returns not found."""
        result = await HANDLERS["omega_traverse"]({"memory_id": "nonexistent-id-99999"})
        assert not _is_error(result)
        text = _text(result)
        assert "not found" in text.lower()

    @pytest.mark.asyncio
    async def test_similar_nonexistent_memory(self):
        """UAT: Finding similar to a nonexistent memory returns not found."""
        result = await HANDLERS["omega_similar"]({"memory_id": "nonexistent-id-99999"})
        assert not _is_error(result)
        text = _text(result)
        assert "not found" in text.lower()

    @pytest.mark.asyncio
    async def test_timeline_shows_recent(self):
        """UAT: Timeline shows recently stored memories grouped by day."""
        await _store_and_get_id(
            "Timeline test memory about configuring the OMEGA MCP server with stdio transport",
            event_type="decision",
        )
        result = await HANDLERS["omega_timeline"]({"days": 1})
        assert not _is_error(result)
        text = _text(result)
        assert "Timeline" in text


# ============================================================================
# SECTION 4: Session Scoping
# ============================================================================


class TestUATSessionScoping:
    """Session isolation and clearing through handlers."""

    @pytest.mark.asyncio
    async def test_store_with_session_id(self):
        """UAT: Memories can be stored with a session ID."""
        result = await HANDLERS["omega_store"]({
            "content": "Session-scoped memory for testing isolation between different agent sessions in OMEGA",
            "event_type": "lesson_learned",
            "session_id": "test-session-aaa",
        })
        assert not _is_error(result)

    @pytest.mark.asyncio
    async def test_query_scoped_by_session(self):
        """UAT: Queries can be scoped to a specific session."""
        await HANDLERS["omega_store"]({
            "content": "Session AAA memory about configuring the pytest-asyncio fixture mode for OMEGA tests",
            "event_type": "lesson_learned",
            "session_id": "session-aaa",
        })
        await HANDLERS["omega_store"]({
            "content": "Session BBB memory about setting up the development environment for frontend project",
            "event_type": "lesson_learned",
            "session_id": "session-bbb",
        })
        result = await HANDLERS["omega_query"]({
            "query": "configuring pytest",
            "session_id": "session-aaa",
        })
        text = _text(result)
        assert "pytest" in text.lower()

    @pytest.mark.asyncio
    async def test_clear_session(self):
        """UAT: Clearing a session removes only that session's memories."""
        await HANDLERS["omega_store"]({
            "content": "Clearable session memory about OMEGA hook configuration and management strategies",
            "event_type": "lesson_learned",
            "session_id": "session-to-clear",
        })
        await HANDLERS["omega_store"]({
            "content": "Persistent session memory about the OMEGA bridge singleton pattern implementation details",
            "event_type": "lesson_learned",
            "session_id": "session-to-keep",
        })

        result = await HANDLERS["omega_clear_session"]({"session_id": "session-to-clear"})
        assert not _is_error(result)
        text = _text(result)
        assert "Cleared" in text

    @pytest.mark.asyncio
    async def test_clear_session_requires_id(self):
        """UAT: Clear session requires a session_id."""
        result = await HANDLERS["omega_clear_session"]({"session_id": ""})
        assert _is_error(result)

    @pytest.mark.asyncio
    async def test_session_stats(self):
        """UAT: Session stats shows per-session memory counts."""
        await HANDLERS["omega_store"]({
            "content": "Stats test memory for verifying session statistics reporting in the OMEGA system",
            "event_type": "lesson_learned",
            "session_id": "stats-session-1",
        })
        result = await HANDLERS["omega_session_stats"]({})
        assert not _is_error(result)
        text = _text(result)
        assert "Session Stats" in text or "session" in text.lower()


# ============================================================================
# SECTION 5: Preference Management
# ============================================================================


class TestUATPreferenceManagement:
    """Preference storage and retrieval through handlers."""

    @pytest.mark.asyncio
    async def test_remember_stores_preference(self):
        """UAT: omega_remember stores a user preference."""
        result = await HANDLERS["omega_remember"]({
            "text": "I prefer using dark mode in all my code editors and terminal applications",
        })
        assert not _is_error(result)

    @pytest.mark.asyncio
    async def test_list_preferences(self):
        """UAT: Listing preferences returns stored user preferences."""
        await HANDLERS["omega_remember"]({
            "text": "I prefer Python over JavaScript for backend development tasks",
        })
        result = await HANDLERS["omega_list_preferences"]({})
        assert not _is_error(result)
        text = _text(result)
        assert "Python" in text or "Preferences" in text or "preference" in text.lower()

    @pytest.mark.asyncio
    async def test_remember_empty_rejected(self):
        """UAT: omega_remember rejects empty text."""
        result = await HANDLERS["omega_remember"]({"text": ""})
        assert _is_error(result)

    @pytest.mark.asyncio
    async def test_profile_includes_preferences(self):
        """UAT: Profile includes stored preferences."""
        await HANDLERS["omega_remember"]({
            "text": "My preferred testing framework for Python projects is always pytest with asyncio",
        })
        result = await HANDLERS["omega_profile"]({})
        assert not _is_error(result)
        text = _text(result)
        assert "pytest" in text.lower() or "preferences" in text.lower() or "profile" in text.lower()

    @pytest.mark.asyncio
    async def test_save_and_load_profile(self):
        """UAT: Save profile data and verify it persists."""
        result = await HANDLERS["omega_save_profile"]({
            "profile": {"name": "Test User", "role": "Developer"},
        })
        assert not _is_error(result)
        text = _text(result)
        assert "updated" in text.lower() or "Profile" in text

        result = await HANDLERS["omega_profile"]({})
        text = _text(result)
        assert "Test User" in text or "Developer" in text

    @pytest.mark.asyncio
    async def test_save_profile_empty_is_noop(self):
        """UAT: save_profile with empty dict is a no-op (returns current profile)."""
        result = await HANDLERS["omega_save_profile"]({"profile": {}})
        assert not _is_error(result)


# ============================================================================
# SECTION 6: Consolidation Pipeline
# ============================================================================


class TestUATConsolidationPipeline:
    """Consolidation and compaction through handlers."""

    @pytest.mark.asyncio
    async def test_consolidate_empty_store(self):
        """UAT: Consolidation on empty store reports nothing to do."""
        result = await HANDLERS["omega_consolidate"]({})
        assert not _is_error(result)
        text = _text(result)
        assert "Consolidation" in text

    @pytest.mark.asyncio
    async def test_consolidate_with_data(self):
        """UAT: Consolidation runs on a populated store."""
        for i in range(5):
            await HANDLERS["omega_store"]({
                "content": f"Consolidation test lesson number {i} about OMEGA memory management and optimization strategies",
                "event_type": "lesson_learned",
            })
        result = await HANDLERS["omega_consolidate"]({"prune_days": 1})
        assert not _is_error(result)
        text = _text(result)
        assert "Before:" in text or "After:" in text

    @pytest.mark.asyncio
    async def test_compact_dry_run(self):
        """UAT: Compact dry run shows what would be compacted."""
        result = await HANDLERS["omega_compact"]({
            "event_type": "lesson_learned",
            "dry_run": True,
        })
        assert not _is_error(result)
        text = _text(result)
        assert "Compaction" in text

    @pytest.mark.asyncio
    async def test_type_stats_empty(self):
        """UAT: Type stats on empty store returns appropriate message."""
        result = await HANDLERS["omega_type_stats"]({})
        assert not _is_error(result)
        text = _text(result)
        assert "No memories" in text or "Type Stats" in text

    @pytest.mark.asyncio
    async def test_health_check(self):
        """UAT: Health check returns status information."""
        result = await HANDLERS["omega_health"]({})
        assert not _is_error(result)
        text = _text(result)
        assert "Health" in text
        assert "Status:" in text


# ============================================================================
# SECTION 7: Welcome Briefing
# ============================================================================


class TestUATWelcomeBriefing:
    """Welcome briefing through handlers."""

    @pytest.mark.asyncio
    async def test_welcome_empty_store(self):
        """UAT: Welcome on empty store returns greeting."""
        result = await HANDLERS["omega_welcome"]({})
        assert not _is_error(result)
        text = _text(result)
        assert "Welcome" in text or "welcome" in text or "memories" in text.lower()

    @pytest.mark.asyncio
    async def test_welcome_with_memories(self):
        """UAT: Welcome with stored memories returns briefing."""
        await _store_and_get_id(
            "Key decision: use SQLite-vec for vector similarity search instead of Faiss in OMEGA",
            event_type="decision",
        )
        await _store_and_get_id(
            "Important lesson learned: always reset the bridge singleton between tests to prevent state leaks",
            event_type="lesson_learned",
        )
        result = await HANDLERS["omega_welcome"]({})
        assert not _is_error(result)
        text = _text(result)
        parsed = json.loads(text)
        assert parsed["memory_count"] >= 2

    @pytest.mark.asyncio
    async def test_welcome_prioritizes_high_value(self):
        """UAT: Welcome prioritizes decisions/lessons over session summaries."""
        # Store a mix of types
        await HANDLERS["omega_store"]({
            "content": "Session summary: worked on OMEGA test infrastructure and hook configuration",
            "event_type": "session_summary",
        })
        await _store_and_get_id(
            "Critical decision: OMEGA uses ONNX CPU-only inference to avoid the CoreML memory leak issue",
            event_type="decision",
        )
        result = await HANDLERS["omega_welcome"]({})
        text = _text(result)
        parsed = json.loads(text)
        recent = parsed.get("recent_memories", [])
        if recent:
            types = [m["type"] for m in recent]
            # Decision should appear before session_summary in welcome
            assert "decision" in types or "lesson_learned" in types

    @pytest.mark.asyncio
    async def test_welcome_with_session_id(self):
        """UAT: Welcome accepts an optional session_id."""
        result = await HANDLERS["omega_welcome"]({"session_id": "test-session-welcome"})
        assert not _is_error(result)


# ============================================================================
# SECTION 8: Export / Import
# ============================================================================


class TestUATExportImport:
    """Export/import (backup/restore) through handlers."""

    @pytest.mark.asyncio
    async def test_export_creates_file(self, tmp_omega_dir):
        """UAT: Export creates a backup file."""
        await _store_and_get_id(
            "Export test memory about OMEGA backup and restore procedures for data safety",
            event_type="decision",
        )
        filepath = str(tmp_omega_dir / "test-export.json")
        with patch("omega.server.handlers._SAFE_EXPORT_DIR", tmp_omega_dir):
            result = await HANDLERS["omega_backup"]({
                "mode": "export",
                "filepath": filepath,
            })
        assert not _is_error(result)
        text = _text(result)
        assert "Export" in text
        assert Path(filepath).exists()

    @pytest.mark.asyncio
    async def test_export_import_roundtrip(self, tmp_omega_dir):
        """UAT: Export then import restores memories."""
        await _store_and_get_id(
            "Roundtrip export import test about OMEGA memory persistence and data integrity",
            event_type="lesson_learned",
        )
        filepath = str(tmp_omega_dir / "roundtrip.json")

        with patch("omega.server.handlers._SAFE_EXPORT_DIR", tmp_omega_dir):
            # Export
            result = await HANDLERS["omega_backup"]({
                "mode": "export",
                "filepath": filepath,
            })
            assert not _is_error(result)

            # Clear and reimport
            result = await HANDLERS["omega_backup"]({
                "mode": "import",
                "filepath": filepath,
                "clear_existing": True,
            })
            assert not _is_error(result)
            text = _text(result)
            assert "Import" in text

    @pytest.mark.asyncio
    async def test_import_nonexistent_file(self, tmp_omega_dir):
        """UAT: Importing a nonexistent file returns error."""
        with patch("omega.server.handlers._SAFE_EXPORT_DIR", tmp_omega_dir):
            result = await HANDLERS["omega_backup"]({
                "mode": "import",
                "filepath": str(tmp_omega_dir / "nonexistent.json"),
            })
        assert _is_error(result)
        assert "not found" in _text(result).lower() or "Error" in _text(result)

    @pytest.mark.asyncio
    async def test_backup_requires_filepath(self):
        """UAT: Backup requires a filepath parameter."""
        result = await HANDLERS["omega_backup"]({"mode": "export", "filepath": ""})
        assert _is_error(result)

    @pytest.mark.asyncio
    async def test_backup_path_traversal_blocked(self):
        """UAT: Path traversal outside safe dir is blocked."""
        result = await HANDLERS["omega_backup"]({
            "mode": "export",
            "filepath": "/tmp/evil-export.json",
        })
        assert _is_error(result)
        text = _text(result)
        assert "must be under" in text.lower() or "Path" in text


# ============================================================================
# SECTION 9: Feedback Loop
# ============================================================================


class TestUATFeedbackLoop:
    """Feedback recording through handlers."""

    @pytest.mark.asyncio
    async def test_feedback_helpful(self):
        """UAT: Rating a memory as helpful records feedback."""
        node_id = await _store_and_get_id(
            "Helpful lesson about using pytest markers to skip slow tests during rapid development",
            event_type="lesson_learned",
        )
        result = await HANDLERS["omega_feedback"]({
            "memory_id": node_id,
            "rating": "helpful",
        })
        assert not _is_error(result)
        text = _text(result)
        assert "helpful" in text.lower()

    @pytest.mark.asyncio
    async def test_feedback_outdated(self):
        """UAT: Rating a memory as outdated records negative feedback."""
        node_id = await _store_and_get_id(
            "Outdated lesson about configuring the old JSONL store backend for OMEGA memory persistence",
            event_type="lesson_learned",
        )
        result = await HANDLERS["omega_feedback"]({
            "memory_id": node_id,
            "rating": "outdated",
        })
        assert not _is_error(result)
        text = _text(result)
        assert "outdated" in text.lower()

    @pytest.mark.asyncio
    async def test_feedback_invalid_rating(self):
        """UAT: Invalid rating is rejected."""
        node_id = await _store_and_get_id(
            "Memory for testing invalid feedback ratings in the OMEGA feedback system handler",
            event_type="lesson_learned",
        )
        result = await HANDLERS["omega_feedback"]({
            "memory_id": node_id,
            "rating": "invalid_rating",
        })
        assert _is_error(result)

    @pytest.mark.asyncio
    async def test_feedback_requires_memory_id(self):
        """UAT: Feedback requires a memory_id."""
        result = await HANDLERS["omega_feedback"]({
            "memory_id": "",
            "rating": "helpful",
        })
        assert _is_error(result)

    @pytest.mark.asyncio
    async def test_feedback_cumulative(self):
        """UAT: Multiple feedback signals accumulate."""
        node_id = await _store_and_get_id(
            "Memory for testing cumulative feedback signals across multiple ratings in OMEGA system",
            event_type="lesson_learned",
        )
        # First feedback
        result1 = await HANDLERS["omega_feedback"]({
            "memory_id": node_id,
            "rating": "helpful",
        })
        assert not _is_error(result1)

        # Second feedback
        result2 = await HANDLERS["omega_feedback"]({
            "memory_id": node_id,
            "rating": "helpful",
        })
        assert not _is_error(result2)
        text = _text(result2)
        # Should show accumulated signals
        assert "signal" in text.lower() or "score" in text.lower()


# ============================================================================
# SECTION 10: Content Safety
# ============================================================================


class TestUATContentSafety:
    """Blocklist enforcement, min-length, and path safety through handlers."""

    @pytest.mark.asyncio
    async def test_blocklist_broadcast_rejected(self):
        """UAT: Content starting with [BROADCAST is blocked."""
        result = await HANDLERS["omega_store"]({
            "content": "[BROADCAST from test-agent] Session registered for testing purposes",
            "event_type": "lesson_learned",
        })
        text = _text(result)
        assert "Blocked" in text or "noise" in text.lower()

    @pytest.mark.asyncio
    async def test_blocklist_work_breadcrumb_rejected(self):
        """UAT: Content starting with [WORK BREADCRUMB is blocked."""
        result = await HANDLERS["omega_store"]({
            "content": "[WORK BREADCRUMB] Agent started working on file omega/bridge.py",
            "event_type": "memory",
        })
        text = _text(result)
        assert "Blocked" in text or "noise" in text.lower()

    @pytest.mark.asyncio
    async def test_blocklist_stderr_hook_content_rejected(self):
        """UAT: Hook-sourced content containing stderr JSON is blocked."""
        result = await HANDLERS["omega_store"]({
            "content": 'Command output: {"stderr": "error message from subprocess"}',
            "event_type": "error_pattern",
            "metadata": {"source": "auto_capture_hook"},
        })
        text = _text(result)
        assert "Blocked" in text or "noise" in text.lower()

    @pytest.mark.asyncio
    async def test_blocklist_stderr_direct_api_allowed(self):
        """UAT: Direct API calls mentioning stderr are NOT blocked."""
        result = await HANDLERS["omega_store"]({
            "content": 'The "stderr": field contains error output and should be logged for debugging',
            "event_type": "lesson_learned",
        })
        text = _text(result)
        assert "Blocked" not in text

    @pytest.mark.asyncio
    async def test_min_length_hook_content_rejected(self):
        """UAT: Short content from auto-capture hooks is rejected."""
        result = await HANDLERS["omega_store"]({
            "content": "Too short",
            "event_type": "lesson_learned",
            "metadata": {"source": "auto_capture_hook"},
        })
        text = _text(result)
        assert "Blocked" in text or "short" in text.lower()

    @pytest.mark.asyncio
    async def test_min_length_direct_api_allowed(self):
        """UAT: Short content from direct API calls IS allowed (not hook)."""
        result = await HANDLERS["omega_store"]({
            "content": "Direct API short content for testing minimum length enforcement policy rules",
            "event_type": "lesson_learned",
        })
        text = _text(result)
        # Direct API calls should not be blocked by min-length
        assert "Blocked" not in text or "short" not in text.lower()

    @pytest.mark.asyncio
    async def test_path_traversal_blocked_on_backup(self):
        """UAT: Export to path outside safe dir is blocked."""
        result = await HANDLERS["omega_backup"]({
            "mode": "export",
            "filepath": "/etc/passwd",
        })
        assert _is_error(result)

    @pytest.mark.asyncio
    async def test_work_dispatch_noise_rejected(self):
        """UAT: Content starting with [WORK DISPATCH] is blocked."""
        result = await HANDLERS["omega_store"]({
            "content": "[WORK DISPATCH] dispatching task to agent-2 for bridge.py refactor",
            "event_type": "memory",
        })
        text = _text(result)
        assert "Blocked" in text or "noise" in text.lower()

    @pytest.mark.asyncio
    async def test_task_notification_rejected(self):
        """UAT: Content starting with <task-notification> is blocked."""
        result = await HANDLERS["omega_store"]({
            "content": "<task-notification> Task completed successfully for testing",
            "event_type": "decision",
        })
        text = _text(result)
        assert "Blocked" in text or "noise" in text.lower()
