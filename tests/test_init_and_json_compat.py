"""Tests for omega.__init__ exports and omega.json_compat wrapper."""
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# ============================================================================
# omega.__init__
# ============================================================================


class TestOmegaInit:
    def test_version_is_string(self):
        import omega
        assert isinstance(omega.__version__, str)
        assert len(omega.__version__) > 0

    def test_all_defined(self):
        import omega
        assert hasattr(omega, "__all__")
        assert len(omega.__all__) > 0

    def test_all_entries_are_importable(self):
        import omega
        for name in omega.__all__:
            assert hasattr(omega, name), f"omega.__all__ lists '{name}' but it's not importable"


# ============================================================================
# omega.json_compat
# ============================================================================


class TestJsonCompatLoads:
    def test_loads_string(self):
        from omega import json_compat as json
        result = json.loads('{"key": "value"}')
        assert result == {"key": "value"}

    def test_loads_bytes(self):
        from omega import json_compat as json
        result = json.loads(b'{"key": 1}')
        assert result == {"key": 1}

    def test_loads_list(self):
        from omega import json_compat as json
        result = json.loads("[1, 2, 3]")
        assert result == [1, 2, 3]


class TestJsonCompatDumps:
    def test_dumps_roundtrip(self):
        from omega import json_compat as json
        obj = {"a": 1, "b": [2, 3], "c": "hello"}
        serialized = json.dumps(obj)
        assert json.loads(serialized) == obj

    def test_dumps_with_indent(self):
        from omega import json_compat as json
        result = json.dumps({"a": 1}, indent=2)
        assert "\n" in result
        assert "  " in result

    def test_dumps_sort_keys(self):
        from omega import json_compat as json
        result = json.dumps({"b": 2, "a": 1}, sort_keys=True)
        assert result.index('"a"') < result.index('"b"')


class TestJsonCompatFileIO:
    def test_load_from_file(self, tmp_path):
        from omega import json_compat as json
        path = tmp_path / "test.json"
        path.write_text('{"data": true}')
        with open(path) as f:
            result = json.load(f)
        assert result == {"data": True}

    def test_dump_to_file(self, tmp_path):
        from omega import json_compat as json
        path = tmp_path / "out.json"
        with open(path, "w") as f:
            json.dump({"x": 42}, f)
        assert json.loads(path.read_text()) == {"x": 42}
