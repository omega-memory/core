"""Tests for OMEGA Phase 3 — Intelligence Layer.

Tests constraint enforcement, cross-project learning, and smart surfacing.
"""
import json
import sys
from pathlib import Path
from unittest.mock import patch

# Ensure omega package is importable
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestConstraintEnforcement:
    """Test per-project constraint loading and matching."""

    def test_no_constraints_dir(self, tmp_omega_dir):
        """No constraints dir → empty results."""
        from omega.bridge import check_constraints
        with patch("omega.bridge.CONSTRAINTS_DIR", tmp_omega_dir / "constraints"):
            result = check_constraints("/some/file.py")
        assert result == []

    def test_global_constraints(self, tmp_omega_dir):
        """Global constraints match files by pattern."""
        from omega.bridge import check_constraints
        cdir = tmp_omega_dir / "constraints"
        cdir.mkdir()
        (cdir / "global.json").write_text(json.dumps({
            "rules": [
                {"pattern": "*.py", "constraint": "Use type hints", "severity": "warn"},
                {"pattern": "*.sql", "constraint": "No DROP TABLE", "severity": "error"},
            ]
        }))

        with patch("omega.bridge.CONSTRAINTS_DIR", cdir):
            matches = check_constraints("/project/main.py")
        assert len(matches) == 1
        assert matches[0]["constraint"] == "Use type hints"
        assert matches[0]["severity"] == "warn"
        assert matches[0]["source"] == "global"

    def test_project_constraints(self, tmp_omega_dir):
        """Project-specific constraints loaded by project name."""
        from omega.bridge import check_constraints
        cdir = tmp_omega_dir / "constraints"
        cdir.mkdir()
        (cdir / "myapp.json").write_text(json.dumps({
            "rules": [
                {"pattern": "*.tsx", "constraint": "Use React.FC", "severity": "warn"},
            ]
        }))

        with patch("omega.bridge.CONSTRAINTS_DIR", cdir):
            matches = check_constraints("/Projects/myapp/Component.tsx", project="/Projects/myapp")
        assert len(matches) == 1
        assert matches[0]["constraint"] == "Use React.FC"
        assert matches[0]["source"] == "myapp"

    def test_merged_global_and_project(self, tmp_omega_dir):
        """Global + project constraints both surface."""
        from omega.bridge import check_constraints
        cdir = tmp_omega_dir / "constraints"
        cdir.mkdir()
        (cdir / "global.json").write_text(json.dumps({
            "rules": [{"pattern": "*.py", "constraint": "Global rule", "severity": "warn"}]
        }))
        (cdir / "omega.json").write_text(json.dumps({
            "rules": [{"pattern": "*.py", "constraint": "Project rule", "severity": "error"}]
        }))

        with patch("omega.bridge.CONSTRAINTS_DIR", cdir):
            matches = check_constraints("/Projects/omega/bridge.py", project="/Projects/omega")
        assert len(matches) == 2
        sources = {m["source"] for m in matches}
        assert sources == {"global", "omega"}

    def test_no_match(self, tmp_omega_dir):
        """Rules that don't match the file return empty."""
        from omega.bridge import check_constraints
        cdir = tmp_omega_dir / "constraints"
        cdir.mkdir()
        (cdir / "global.json").write_text(json.dumps({
            "rules": [{"pattern": "*.sql", "constraint": "SQL rule", "severity": "warn"}]
        }))

        with patch("omega.bridge.CONSTRAINTS_DIR", cdir):
            matches = check_constraints("/project/main.py")
        assert matches == []

    def test_list_constraints(self, tmp_omega_dir):
        """list_constraints returns all rules."""
        from omega.bridge import list_constraints
        cdir = tmp_omega_dir / "constraints"
        cdir.mkdir()
        (cdir / "global.json").write_text(json.dumps({
            "rules": [
                {"pattern": "*.py", "constraint": "Rule 1", "severity": "warn"},
                {"pattern": "*.js", "constraint": "Rule 2", "severity": "error"},
            ]
        }))

        with patch("omega.bridge.CONSTRAINTS_DIR", cdir):
            info = list_constraints()
        assert info["count"] == 2
        assert len(info["rules"]) == 2

    def test_save_constraints(self, tmp_omega_dir):
        """save_constraints writes rules to disk."""
        from omega.bridge import save_constraints, list_constraints
        cdir = tmp_omega_dir / "constraints"

        with patch("omega.bridge.CONSTRAINTS_DIR", cdir):
            result = save_constraints(
                [{"pattern": "*.py", "constraint": "Test rule", "severity": "warn"}],
                project="/Projects/testproj",
            )
        assert result["count"] == 1
        assert "testproj.json" in result["saved"]
        assert (cdir / "testproj.json").exists()

        # Verify round-trip
        with patch("omega.bridge.CONSTRAINTS_DIR", cdir):
            info = list_constraints(project="/Projects/testproj")
        assert info["count"] == 1
        assert info["rules"][0]["constraint"] == "Test rule"

    def test_save_global_constraints(self, tmp_omega_dir):
        """save_constraints without project saves to global.json."""
        from omega.bridge import save_constraints
        cdir = tmp_omega_dir / "constraints"

        with patch("omega.bridge.CONSTRAINTS_DIR", cdir):
            result = save_constraints(
                [{"pattern": "*.md", "constraint": "No TODOs", "severity": "warn"}],
            )
        assert "global.json" in result["saved"]

    def test_malformed_json_ignored(self, tmp_omega_dir):
        """Malformed constraint files don't crash, return empty."""
        from omega.bridge import check_constraints
        cdir = tmp_omega_dir / "constraints"
        cdir.mkdir()
        (cdir / "global.json").write_text("not valid json{{{")

        with patch("omega.bridge.CONSTRAINTS_DIR", cdir):
            matches = check_constraints("/project/main.py")
        assert matches == []


class TestCrossProjectLessons:
    """Test cross-project lesson retrieval."""

    def test_cross_project_basic(self, store):
        """Lessons from different projects surface with project counts."""
        from omega.bridge import get_cross_project_lessons

        # Store lessons from different projects
        store.store(
            content="Always validate inputs at API boundaries",
            metadata={
                "event_type": "lesson_learned",
                "project": "/Projects/alpha",
                "session_id": "s1",
            },
        )
        store.store(
            content="Use structured logging not print statements",
            metadata={
                "event_type": "lesson_learned",
                "project": "/Projects/beta",
                "session_id": "s2",
            },
        )

        with patch("omega.bridge._get_store", return_value=store):
            lessons = get_cross_project_lessons(limit=10)

        assert len(lessons) >= 1
        # All should have projects_seen field
        for lesson in lessons:
            assert "projects_seen" in lesson
            assert "cross_project" in lesson

    def test_exclude_project(self, store):
        """exclude_project filters out lessons from that project."""
        from omega.bridge import get_cross_project_lessons

        store.store(
            content="Lesson from alpha project",
            metadata={
                "event_type": "lesson_learned",
                "project": "/Projects/alpha",
                "session_id": "s1",
            },
        )
        store.store(
            content="Lesson from beta project",
            metadata={
                "event_type": "lesson_learned",
                "project": "/Projects/beta",
                "session_id": "s2",
            },
        )

        with patch("omega.bridge._get_store", return_value=store):
            lessons = get_cross_project_lessons(exclude_project="/Projects/alpha")

        for lesson in lessons:
            assert lesson.get("source_project") != "/Projects/alpha"

    def test_empty_when_no_lessons(self, store):
        """Returns empty list when no lessons exist."""
        from omega.bridge import get_cross_project_lessons

        with patch("omega.bridge._get_store", return_value=store):
            lessons = get_cross_project_lessons()

        assert lessons == []

    def test_task_filter(self, store):
        """Task filter narrows lesson search."""
        from omega.bridge import get_cross_project_lessons

        store.store(
            content="Use connection pooling for database access",
            metadata={
                "event_type": "lesson_learned",
                "project": "/Projects/alpha",
                "session_id": "s1",
            },
        )

        with patch("omega.bridge._get_store", return_value=store):
            lessons = get_cross_project_lessons(task="database optimization")

        # Should still return results (task is just a search hint)
        assert isinstance(lessons, list)


class TestMCPHandlers:
    """Test MCP handler registration and schemas."""

    def test_handler_count(self):
        """Verify handlers include expected tools."""
        from omega.server.handlers import HANDLERS
        assert "omega_lessons" in HANDLERS  # merged cross_project_lessons into lessons
        assert "omega_backup" in HANDLERS  # merged export+import into backup
        assert len(HANDLERS) == 28  # 24 base + 3 reminder tools + 1 weekly_digest

    def test_schema_count(self):
        """Verify schemas include expected tools."""
        from omega.server.tool_schemas import TOOL_SCHEMAS
        schema_names = {s["name"] for s in TOOL_SCHEMAS}
        assert "omega_lessons" in schema_names
        assert "omega_backup" in schema_names
        assert "omega_weekly_digest" in schema_names
        assert len(TOOL_SCHEMAS) == 25  # 24 base + weekly_digest

    def test_handler_schema_parity(self):
        """Every schema has a matching handler."""
        from omega.server.handlers import HANDLERS
        from omega.server.tool_schemas import TOOL_SCHEMAS
        schema_names = {s["name"] for s in TOOL_SCHEMAS}
        for name in schema_names:
            assert name in HANDLERS, f"Schema {name} has no handler"
