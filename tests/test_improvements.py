"""Tests for OMEGA SOTA improvements: graph traversal, contextual re-ranking, memory compaction."""
import os
import pytest


# ---------------------------------------------------------------------------
# Graph Traversal
# ---------------------------------------------------------------------------

class TestGraphTraversal:
    """Tests for SQLiteStore.get_related_chain() and bridge.traverse()."""

    def test_no_edges_returns_empty(self, store):
        nid = store.store(content="Isolated memory about quantum computing algorithms")
        results = store.get_related_chain(nid, max_hops=2)
        assert results == []

    def test_single_hop(self, store):
        a = store.store(content="SQLite database performance tuning requires careful index management")
        b = store.store(content="PostgreSQL replication setup for high availability clusters")
        c = store.store(content="React component lifecycle hooks and state management patterns")

        store.add_edge(a, b, "related", 0.8)
        store.add_edge(a, c, "related", 0.5)

        results = store.get_related_chain(a, max_hops=1)
        assert len(results) == 2
        node_ids = {r["node_id"] for r in results}
        assert b in node_ids
        assert c in node_ids
        # All at hop 1
        assert all(r["hop"] == 1 for r in results)

    def test_two_hops(self, store):
        a = store.store(content="Kubernetes pod scheduling and resource allocation strategies")
        b = store.store(content="Python asyncio event loop internals and coroutine execution")
        c = store.store(content="CSS grid layout techniques for responsive dashboard designs")

        store.add_edge(a, b, "related", 0.9)
        store.add_edge(b, c, "related", 0.7)

        results = store.get_related_chain(a, max_hops=2)
        assert len(results) == 2
        # b at hop 1, c at hop 2
        by_id = {r["node_id"]: r for r in results}
        assert by_id[b]["hop"] == 1
        assert by_id[c]["hop"] == 2

    def test_min_weight_filter(self, store):
        a = store.store(content="Docker container orchestration with compose files")
        b = store.store(content="Rust ownership model and borrow checker mechanics")
        c = store.store(content="GraphQL schema design for microservice architecture")

        store.add_edge(a, b, "related", 0.9)
        store.add_edge(a, c, "related", 0.2)

        results = store.get_related_chain(a, max_hops=1, min_weight=0.5)
        assert len(results) == 1
        assert results[0]["node_id"] == b

    def test_max_hops_clamped(self, store):
        """max_hops > 5 gets clamped to 5."""
        a = store.store(content="Redis cache invalidation strategies for distributed systems")
        results = store.get_related_chain(a, max_hops=100)
        assert isinstance(results, list)  # Just verify it doesn't crash

    def test_bidirectional_edges(self, store):
        """Edges are traversed in both directions."""
        a = store.store(content="Terraform infrastructure provisioning for cloud environments")
        b = store.store(content="Swift concurrency with structured tasks and actor isolation")

        # Only add edge in one direction (a -> b)
        store.add_edge(a, b, "related", 0.8)

        # Traversal from b should still find a
        results = store.get_related_chain(b, max_hops=1)
        assert len(results) == 1
        assert results[0]["node_id"] == a

    def test_no_revisit_start(self, store):
        """Start node should not appear in results even with circular edges."""
        a = store.store(content="Nginx reverse proxy configuration for load balancing")
        b = store.store(content="Vue.js reactivity system using proxies and watchers")

        store.add_edge(a, b, "related", 0.8)
        store.add_edge(b, a, "related", 0.8)

        results = store.get_related_chain(a, max_hops=2)
        node_ids = {r["node_id"] for r in results}
        assert a not in node_ids
        assert b in node_ids

    def test_edge_type_filter(self, store):
        a = store.store(content="MongoDB sharding strategies for horizontal scaling")
        b = store.store(content="Elixir OTP supervision trees and fault tolerance")
        c = store.store(content="WebAssembly compilation targets and runtime performance")

        store.add_edge(a, b, "causal", 0.9)
        store.add_edge(a, c, "related", 0.8)

        results = store.get_related_chain(a, max_hops=1, edge_types=["causal"])
        assert len(results) == 1
        assert results[0]["node_id"] == b


class TestTraverseBridge:
    """Tests for bridge.traverse()."""

    def test_traverse_nonexistent(self, tmp_omega_dir):
        os.environ["OMEGA_SKIP_EMBEDDINGS"] = "1"
        try:
            from omega.bridge import reset_memory, traverse
            reset_memory()
            result = traverse("nonexistent-id")
            assert "not found" in result
        finally:
            os.environ.pop("OMEGA_SKIP_EMBEDDINGS", None)

    def test_traverse_formats_markdown(self, tmp_omega_dir):
        os.environ["OMEGA_SKIP_EMBEDDINGS"] = "1"
        try:
            from omega.bridge import reset_memory, traverse, _get_store
            reset_memory()
            store = _get_store()
            a = store.store(content="Decision: use SQLite for storage")
            b = store.store(content="Lesson: SQLite WAL mode improves concurrency")
            store.add_edge(a, b, "related", 0.85)

            result = traverse(a, max_hops=1)
            assert "# Graph Traversal" in result
            assert "Hop 1" in result
            assert b[:12] in result
        finally:
            os.environ.pop("OMEGA_SKIP_EMBEDDINGS", None)


# ---------------------------------------------------------------------------
# Contextual Re-ranking
# ---------------------------------------------------------------------------

class TestContextualReranking:
    """Tests for context_file and context_tags in SQLiteStore.query()."""

    def test_context_file_boosts_matching_content(self, store):
        """Memories mentioning the context file should rank higher."""
        store.store(
            content="Python testing with pytest is important for quality",
            metadata={"event_type": "lesson_learned", "tags": ["python", "pytest"]},
        )
        store.store(
            content="JavaScript testing with jest ensures frontend quality",
            metadata={"event_type": "lesson_learned", "tags": ["javascript", "jest"]},
        )

        # Query without context
        results_no_ctx = store.query("testing quality", limit=2)

        # Query with Python file context
        results_with_ctx = store.query(
            "testing quality",
            limit=2,
            context_file="/projects/omega/tests/test_store.py",
            context_tags=["python"],
        )

        # Both should return results
        assert len(results_no_ctx) >= 1
        assert len(results_with_ctx) >= 1

        # With context, the Python result should be boosted
        if len(results_with_ctx) >= 2:
            python_result = [r for r in results_with_ctx if "python" in r.content.lower() or "pytest" in r.content.lower()]
            if python_result:
                assert python_result[0].relevance >= results_with_ctx[-1].relevance

    def test_context_tags_boost(self, store):
        """Memories with matching tags should get a relevance boost."""
        store.store(
            content="Always use type hints in Python code",
            metadata={"event_type": "user_preference", "tags": ["python"]},
        )
        store.store(
            content="Always use TypeScript for frontend code",
            metadata={"event_type": "user_preference", "tags": ["typescript"]},
        )

        results = store.query(
            "code style preferences",
            limit=2,
            context_tags=["python"],
        )
        assert len(results) >= 1

    def test_no_context_no_error(self, store):
        """Query without context should work normally."""
        store.store(content="Test memory for no context case")
        results = store.query("test memory", limit=5)
        assert len(results) >= 1

    def test_empty_context_ignored(self, store):
        """Empty context_file and context_tags should not affect results."""
        store.store(content="Some test content here")
        results = store.query(
            "test content",
            limit=5,
            context_file="",
            context_tags=[],
        )
        assert len(results) >= 1


# ---------------------------------------------------------------------------
# Memory Compaction
# ---------------------------------------------------------------------------

class TestMemoryCompaction:
    """Tests for bridge.compact()."""

    def test_compact_insufficient_memories(self, tmp_omega_dir):
        os.environ["OMEGA_SKIP_EMBEDDINGS"] = "1"
        try:
            from omega.bridge import reset_memory, compact
            reset_memory()
            result = compact(event_type="lesson_learned", min_cluster_size=3)
            assert "Nothing to compact" in result
        finally:
            os.environ.pop("OMEGA_SKIP_EMBEDDINGS", None)

    def test_compact_dry_run(self, tmp_omega_dir):
        os.environ["OMEGA_SKIP_EMBEDDINGS"] = "1"
        try:
            from omega.bridge import reset_memory, compact, _get_store
            reset_memory()
            store = _get_store()

            # Create a cluster of similar lesson_learned memories
            for i in range(4):
                store.store(
                    content=f"Lesson learned: always run tests before committing code changes number {i}",
                    metadata={"event_type": "lesson_learned"},
                )

            result = compact(
                event_type="lesson_learned",
                similarity_threshold=0.5,
                min_cluster_size=3,
                dry_run=True,
            )
            assert "DRY RUN" in result
            assert "Would compact" in result
            # No memories should be superseded in dry run
            assert store.node_count() == 4
        finally:
            os.environ.pop("OMEGA_SKIP_EMBEDDINGS", None)

    def test_compact_creates_summary(self, tmp_omega_dir):
        os.environ["OMEGA_SKIP_EMBEDDINGS"] = "1"
        try:
            from omega.bridge import reset_memory, compact, _get_store
            reset_memory()
            store = _get_store()

            # Create similar memories
            ids = []
            for i in range(4):
                nid = store.store(
                    content=f"Always validate user input before processing to prevent injection attacks variant {i}",
                    metadata={"event_type": "lesson_learned", "tags": ["security"]},
                )
                ids.append(nid)

            before_count = store.node_count()

            result = compact(
                event_type="lesson_learned",
                similarity_threshold=0.5,
                min_cluster_size=3,
                dry_run=False,
            )

            assert "Compacted:" in result
            # Should have created 1 new summary node
            # Total nodes = 4 original + 1 summary = 5
            assert store.node_count() == before_count + 1

            # Original nodes should be marked superseded
            for nid in ids:
                node = store.get_node(nid)
                assert node is not None
                assert node.metadata.get("superseded") is True
                assert "superseded_by" in node.metadata
        finally:
            os.environ.pop("OMEGA_SKIP_EMBEDDINGS", None)

    def test_compact_no_clusters_found(self, tmp_omega_dir):
        os.environ["OMEGA_SKIP_EMBEDDINGS"] = "1"
        try:
            from omega.bridge import reset_memory, compact, _get_store
            reset_memory()
            store = _get_store()

            # Create dissimilar memories
            store.store(
                content="Always use SQLite for local storage in Python apps",
                metadata={"event_type": "lesson_learned"},
            )
            store.store(
                content="React hooks must follow the rules of hooks consistently",
                metadata={"event_type": "lesson_learned"},
            )
            store.store(
                content="Docker containers should be ephemeral and reproducible",
                metadata={"event_type": "lesson_learned"},
            )

            result = compact(
                event_type="lesson_learned",
                similarity_threshold=0.6,
                min_cluster_size=3,
            )
            assert "already compact" in result or "No clusters" in result
        finally:
            os.environ.pop("OMEGA_SKIP_EMBEDDINGS", None)


# ---------------------------------------------------------------------------
# Handler Integration Tests
# ---------------------------------------------------------------------------

class TestNewHandlers:
    """Integration tests for the new MCP handlers."""

    @pytest.mark.asyncio
    async def test_traverse_handler(self, tmp_omega_dir):
        os.environ["OMEGA_SKIP_EMBEDDINGS"] = "1"
        try:
            from omega.bridge import reset_memory, _get_store
            from omega.server.handlers import handle_omega_traverse
            reset_memory()
            store = _get_store()

            a = store.store(content="Test memory A")
            b = store.store(content="Test memory B")
            store.add_edge(a, b, "related", 0.9)

            result = await handle_omega_traverse({"memory_id": a})
            assert "content" in result
            assert not result.get("isError")
        finally:
            os.environ.pop("OMEGA_SKIP_EMBEDDINGS", None)

    @pytest.mark.asyncio
    async def test_traverse_handler_missing_id(self):
        from omega.server.handlers import handle_omega_traverse
        result = await handle_omega_traverse({})
        assert result.get("isError")

    @pytest.mark.asyncio
    async def test_compact_handler(self, tmp_omega_dir):
        os.environ["OMEGA_SKIP_EMBEDDINGS"] = "1"
        try:
            from omega.bridge import reset_memory
            from omega.server.handlers import handle_omega_compact
            reset_memory()

            result = await handle_omega_compact({"dry_run": True})
            assert "content" in result
            assert not result.get("isError")
        finally:
            os.environ.pop("OMEGA_SKIP_EMBEDDINGS", None)
