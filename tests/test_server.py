"""OMEGA MCP Server integration tests — full handler coverage."""
import pytest
from omega.server.tool_schemas import TOOL_SCHEMAS
from omega.server.handlers import HANDLERS


# ============================================================================
# Schema / Registry Tests
# ============================================================================

def test_all_tools_have_handlers():
    """Every tool in TOOL_SCHEMAS should have a handler."""
    for schema in TOOL_SCHEMAS:
        assert schema["name"] in HANDLERS, f"Missing handler for {schema['name']}"

def test_tool_schemas_valid():
    """All tool schemas should have required fields."""
    for schema in TOOL_SCHEMAS:
        assert "name" in schema
        assert "description" in schema
        assert "inputSchema" in schema
        assert schema["name"].startswith("omega_")

def test_handler_count():
    """Should have a handler for every tool schema (plus backward-compat aliases)."""
    schema_names = {s["name"] for s in TOOL_SCHEMAS}
    # Every schema must have a handler
    for name in schema_names:
        assert name in HANDLERS, f"Missing handler for schema: {name}"
    # Handlers may have aliases (backward compat) so len(HANDLERS) >= len(TOOL_SCHEMAS)
    assert len(HANDLERS) >= len(TOOL_SCHEMAS)
    assert len(TOOL_SCHEMAS) == 25  # 24 base + weekly_digest


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
# Helper to store a test memory and return its node_id
# ============================================================================

async def _store_test_memory(content="Test memory for handler tests", event_type="lesson_learned"):
    """Store a memory via the handler, return the node_id from the response."""
    result = await HANDLERS["omega_store"]({"content": content, "event_type": event_type})
    assert not result.get("isError"), result
    # Extract node_id from the markdown response
    text = result["content"][0]["text"]
    for line in text.splitlines():
        if "Node ID" in line and "`" in line:
            return line.split("`")[1]
    return None


# ============================================================================
# Handler: omega_health (merged from omega_status)
# ============================================================================

@pytest.mark.asyncio
async def test_omega_health_handler():
    """Test the health handler returns valid response."""
    result = await HANDLERS["omega_health"]({})
    assert "content" in result
    assert not result.get("isError")


# ============================================================================
# Handler: omega_store + omega_query (existing)
# ============================================================================

@pytest.mark.asyncio
async def test_omega_store_and_query():
    """Test storing and querying a memory."""
    result = await HANDLERS["omega_store"]({
        "content": "Test memory for integration test",
        "event_type": "lesson_learned",
    })
    assert "content" in result
    assert not result.get("isError")

    result = await HANDLERS["omega_query"]({
        "query": "integration test",
        "limit": 5,
    })
    assert "content" in result
    assert not result.get("isError")


# ============================================================================
# Handler: omega_remember
# ============================================================================

@pytest.mark.asyncio
async def test_omega_remember():
    """Remembering stores a user_preference memory."""
    result = await HANDLERS["omega_remember"]({"text": "I prefer dark mode"})
    assert not result.get("isError")
    text = result["content"][0]["text"]
    assert "Memory Captured" in text or "Memory" in text

@pytest.mark.asyncio
async def test_omega_remember_empty():
    """Empty text should return an error."""
    result = await HANDLERS["omega_remember"]({"text": ""})
    assert result.get("isError")


# ============================================================================
# Handler: omega_welcome
# ============================================================================

@pytest.mark.asyncio
async def test_omega_welcome():
    """Welcome should return a JSON briefing."""
    result = await HANDLERS["omega_welcome"]({})
    assert not result.get("isError")
    text = result["content"][0]["text"]
    assert "greeting" in text
    assert "memory_count" in text

@pytest.mark.asyncio
async def test_omega_welcome_with_project():
    """Welcome accepts an optional project parameter."""
    result = await HANDLERS["omega_welcome"]({"project": "/tmp/testproject"})
    assert not result.get("isError")


# ============================================================================
# Handler: omega_profile
# ============================================================================

@pytest.mark.asyncio
async def test_omega_profile():
    """Profile handler returns without error."""
    result = await HANDLERS["omega_profile"]({})
    assert not result.get("isError")
    text = result["content"][0]["text"]
    # May be empty ("No profile") or contain data — either is valid
    assert len(text) > 0


# ============================================================================
# Handler: omega_save_profile
# ============================================================================

@pytest.mark.asyncio
async def test_omega_save_profile():
    """Saving profile should succeed and persist."""
    result = await HANDLERS["omega_save_profile"]({
        "profile": {"name": "Test User", "timezone": "UTC"},
    })
    assert not result.get("isError")
    text = result["content"][0]["text"]
    assert "2 field" in text

    # Verify it persisted
    result = await HANDLERS["omega_profile"]({})
    assert not result.get("isError")
    text = result["content"][0]["text"]
    assert "Test User" in text

@pytest.mark.asyncio
async def test_omega_save_profile_empty():
    """Empty profile dict = read mode (no update), should succeed."""
    result = await HANDLERS["omega_save_profile"]({"profile": {}})
    assert not result.get("isError")  # Reads profile instead of erroring


# ============================================================================
# Handler: omega_delete_memory
# ============================================================================

@pytest.mark.asyncio
async def test_omega_delete_memory():
    """Store then delete a memory."""
    node_id = await _store_test_memory("Memory to delete")
    assert node_id

    result = await HANDLERS["omega_delete_memory"]({"memory_id": node_id})
    assert not result.get("isError")
    text = result["content"][0]["text"]
    assert "Deleted" in text

@pytest.mark.asyncio
async def test_omega_delete_memory_not_found():
    """Deleting a nonexistent memory should return an error."""
    result = await HANDLERS["omega_delete_memory"]({"memory_id": "mem-nonexistent"})
    assert result.get("isError")

@pytest.mark.asyncio
async def test_omega_delete_memory_empty_id():
    """Empty memory_id should return an error."""
    result = await HANDLERS["omega_delete_memory"]({"memory_id": ""})
    assert result.get("isError")


# ============================================================================
# Handler: omega_edit_memory
# ============================================================================

@pytest.mark.asyncio
async def test_omega_edit_memory():
    """Store then edit a memory."""
    node_id = await _store_test_memory("Original content")
    assert node_id

    result = await HANDLERS["omega_edit_memory"]({
        "memory_id": node_id,
        "new_content": "Updated content",
    })
    assert not result.get("isError")
    text = result["content"][0]["text"]
    assert "Updated" in text

@pytest.mark.asyncio
async def test_omega_edit_memory_not_found():
    """Editing a nonexistent memory should return an error."""
    result = await HANDLERS["omega_edit_memory"]({
        "memory_id": "mem-nonexistent",
        "new_content": "new stuff",
    })
    assert result.get("isError")

@pytest.mark.asyncio
async def test_omega_edit_memory_empty_fields():
    """Missing required fields should return errors."""
    result = await HANDLERS["omega_edit_memory"]({"memory_id": "", "new_content": "x"})
    assert result.get("isError")

    result = await HANDLERS["omega_edit_memory"]({"memory_id": "x", "new_content": ""})
    assert result.get("isError")


# ============================================================================
# Handler: omega_list_preferences
# ============================================================================

@pytest.mark.asyncio
async def test_omega_list_preferences_empty():
    """No preferences stored yet returns a message."""
    result = await HANDLERS["omega_list_preferences"]({})
    assert not result.get("isError")
    text = result["content"][0]["text"]
    assert "No preferences" in text or "Preferences" in text

@pytest.mark.asyncio
async def test_omega_list_preferences_after_remember():
    """After remembering a preference, list should show it."""
    await HANDLERS["omega_remember"]({"text": "I use tabs not spaces"})
    result = await HANDLERS["omega_list_preferences"]({})
    assert not result.get("isError")
    text = result["content"][0]["text"]
    assert "tabs" in text or "Preferences" in text


# ============================================================================
# Handler: omega_health
# ============================================================================

@pytest.mark.asyncio
async def test_omega_health():
    """Health check should return formatted markdown."""
    result = await HANDLERS["omega_health"]({})
    assert not result.get("isError")
    text = result["content"][0]["text"]
    assert "OMEGA Health" in text
    assert "Status" in text

@pytest.mark.asyncio
async def test_omega_health_custom_thresholds():
    """Health check accepts custom threshold parameters."""
    result = await HANDLERS["omega_health"]({
        "warn_mb": 50, "critical_mb": 200, "max_nodes": 5000,
    })
    assert not result.get("isError")


# ============================================================================
# Handler: omega_lessons
# ============================================================================

@pytest.mark.asyncio
async def test_omega_lessons_empty():
    """No lessons yet returns a helpful message."""
    result = await HANDLERS["omega_lessons"]({})
    assert not result.get("isError")
    text = result["content"][0]["text"]
    assert "No cross-session lessons" in text or "Lessons" in text

@pytest.mark.asyncio
async def test_omega_lessons_with_data():
    """After storing lessons, the handler should return them."""
    await HANDLERS["omega_store"]({
        "content": "Always run tests before committing code",
        "event_type": "lesson_learned",
    })
    result = await HANDLERS["omega_lessons"]({"limit": 5})
    assert not result.get("isError")
    text = result["content"][0]["text"]
    assert "tests" in text.lower() or "Lessons" in text


# ============================================================================
# Handler: omega_feedback
# ============================================================================

@pytest.mark.asyncio
async def test_omega_feedback():
    """Record feedback on a stored memory."""
    node_id = await _store_test_memory("Feedback target memory")
    assert node_id

    result = await HANDLERS["omega_feedback"]({
        "memory_id": node_id,
        "rating": "helpful",
        "reason": "Very useful",
    })
    assert not result.get("isError")
    text = result["content"][0]["text"]
    assert "Feedback recorded" in text
    assert "helpful" in text

@pytest.mark.asyncio
async def test_omega_feedback_invalid_rating():
    """Invalid rating should return an error."""
    result = await HANDLERS["omega_feedback"]({
        "memory_id": "mem-fake",
        "rating": "amazing",
    })
    assert result.get("isError")
    assert "must be one of" in result["content"][0]["text"]

@pytest.mark.asyncio
async def test_omega_feedback_missing_fields():
    """Missing required fields should return errors."""
    result = await HANDLERS["omega_feedback"]({"memory_id": "", "rating": "helpful"})
    assert result.get("isError")

    result = await HANDLERS["omega_feedback"]({"memory_id": "x", "rating": ""})
    assert result.get("isError")


# ============================================================================
# Handler: omega_clear_session
# ============================================================================

@pytest.mark.asyncio
async def test_omega_clear_session():
    """Clear session should remove all memories for that session."""
    await HANDLERS["omega_store"]({
        "content": "Session-scoped memory",
        "event_type": "memory",
        "session_id": "test-sess-123",
    })
    result = await HANDLERS["omega_clear_session"]({"session_id": "test-sess-123"})
    assert not result.get("isError")
    text = result["content"][0]["text"]
    assert "Cleared" in text

@pytest.mark.asyncio
async def test_omega_clear_session_empty():
    """Empty session_id should return an error."""
    result = await HANDLERS["omega_clear_session"]({"session_id": ""})
    assert result.get("isError")


# ============================================================================
# Handler: omega_query input validation
# ============================================================================

@pytest.mark.asyncio
async def test_omega_query_empty():
    """Empty query should return an error."""
    result = await HANDLERS["omega_query"]({"query": ""})
    assert result.get("isError")


# ============================================================================
# Handler: omega_store input validation
# ============================================================================

@pytest.mark.asyncio
async def test_omega_store_empty():
    """Empty content should return an error."""
    result = await HANDLERS["omega_store"]({"content": ""})
    assert result.get("isError")


# ============================================================================
# Handler: omega_similar
# ============================================================================

@pytest.mark.asyncio
async def test_omega_similar():
    """Find memories similar to a stored memory."""
    node_id = await _store_test_memory("Memory about Python testing")
    assert node_id
    result = await HANDLERS["omega_similar"]({"memory_id": node_id, "limit": 3})
    assert not result.get("isError")
    assert "Similar" in result["content"][0]["text"]

@pytest.mark.asyncio
async def test_omega_similar_not_found():
    """Nonexistent memory_id returns a not-found message (not an error)."""
    result = await HANDLERS["omega_similar"]({"memory_id": "mem-nonexistent"})
    assert not result.get("isError")
    assert "not found" in result["content"][0]["text"]


# ============================================================================
# Handler: omega_timeline
# ============================================================================

@pytest.mark.asyncio
async def test_omega_timeline():
    """Timeline should show recently stored memories."""
    await _store_test_memory("Timeline test memory")
    result = await HANDLERS["omega_timeline"]({"days": 7})
    assert not result.get("isError")
    assert "Timeline" in result["content"][0]["text"]

@pytest.mark.asyncio
async def test_omega_timeline_empty():
    """Timeline with 0 days should return an empty result."""
    result = await HANDLERS["omega_timeline"]({"days": 0})
    assert not result.get("isError")


# ============================================================================
# Handler: omega_consolidate
# ============================================================================

@pytest.mark.asyncio
async def test_omega_consolidate_empty():
    """Consolidation on empty store returns clean report."""
    result = await HANDLERS["omega_consolidate"]({})
    assert not result.get("isError")
    text = result["content"][0]["text"]
    assert "Consolidation" in text
    assert "Nothing to consolidate" in text or "Removed" in text

@pytest.mark.asyncio
async def test_omega_consolidate_with_data():
    """Consolidation with data returns a breakdown."""
    await _store_test_memory("Consolidation test memory")
    result = await HANDLERS["omega_consolidate"]({"prune_days": 30, "max_summaries": 50})
    assert not result.get("isError")
    text = result["content"][0]["text"]
    assert "Consolidation" in text
    assert "Before" in text
    assert "After" in text


# ============================================================================
# Handler: omega_backup — expanduser
# ============================================================================

def test_omega_backup_tilde_expansion():
    """Backup path resolution should expand ~ to home directory."""
    from pathlib import Path

    # Simulate what the handler does: expanduser + resolve
    tilde_path = "~/.omega/test_backup.json"
    resolved = Path(tilde_path).expanduser().resolve()
    home = Path.home().resolve()
    safe_dir = (home / ".omega").resolve()

    # ~ should expand to real home, not stay literal
    assert str(resolved).startswith(str(safe_dir))
    assert "~" not in str(resolved)


# ============================================================================
# Handler: _clamp_int consistency
# ============================================================================

@pytest.mark.asyncio
async def test_similar_clamps_limit():
    """omega_similar should clamp limit to safe bounds."""
    node_id = await _store_test_memory("Clamp test memory")
    assert node_id
    # Negative limit should be clamped to min (1)
    result = await HANDLERS["omega_similar"]({"memory_id": node_id, "limit": -5})
    assert not result.get("isError")

@pytest.mark.asyncio
async def test_timeline_clamps_days():
    """omega_timeline should clamp days to safe bounds."""
    result = await HANDLERS["omega_timeline"]({"days": 99999})
    assert not result.get("isError")

@pytest.mark.asyncio
async def test_compact_clamps_min_cluster_size():
    """omega_compact should clamp min_cluster_size to safe bounds."""
    result = await HANDLERS["omega_compact"]({"min_cluster_size": -1, "dry_run": True})
    assert not result.get("isError")


# ============================================================================
# Schema / docstring accuracy
# ============================================================================

def test_tool_schemas_docstring_count():
    """tool_schemas.py docstring should match actual schema count."""
    import omega.server.tool_schemas as ts
    import inspect
    source = inspect.getsource(ts)
    # Docstring says "25 tools"
    assert "25 tools" in source
    assert len(TOOL_SCHEMAS) == 25  # 24 base + weekly_digest
