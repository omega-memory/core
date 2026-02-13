"""Tests for OMEGA graphs/embeddings module — covers public API and internal helpers."""

import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import omega.graphs as graphs
from omega.graphs import (
    get_embedding_model_info,
    reset_embedding_state,
    _maybe_unload_model,
    _check_onnx_runtime,
    _check_sentence_transformers,
    _hash_embedding,
    generate_embedding,
    generate_embeddings_batch,
    _get_onnx_model_dir,
    preload_embedding_model,
    _has_embedding_backend,
    get_active_backend,
)


# ---------------------------------------------------------------------------
# 1. get_embedding_model_info
# ---------------------------------------------------------------------------

class TestGetEmbeddingModelInfo:
    """Test get_embedding_model_info() returns correct defaults and structure."""

    def test_default_values(self):
        info = get_embedding_model_info()
        assert info["model_name"] == "bge-small-en-v1.5"
        assert info["model_version"] == "v1.5"
        assert info["model_loaded"] is False
        assert info["backend"] is None

    def test_dict_keys(self):
        info = get_embedding_model_info()
        expected_keys = {"model_name", "model_version", "model_loaded", "backend"}
        assert set(info.keys()) == expected_keys

    def test_reflects_backend_state(self, monkeypatch):
        """If backend globals are modified, info should reflect them."""
        monkeypatch.setattr(graphs, "_EMBEDDING_BACKEND", "onnx")
        monkeypatch.setattr(graphs, "_EMBEDDING_MODEL", "fake-model-object")
        info = get_embedding_model_info()
        assert info["model_loaded"] is True
        assert info["backend"] == "onnx"


# ---------------------------------------------------------------------------
# 2. reset_embedding_state
# ---------------------------------------------------------------------------

class TestResetEmbeddingState:
    """Test reset_embedding_state() clears all globals back to defaults."""

    def test_resets_model_and_backend(self, monkeypatch):
        monkeypatch.setattr(graphs, "_EMBEDDING_MODEL", "something")
        monkeypatch.setattr(graphs, "_EMBEDDING_BACKEND", "onnx")
        monkeypatch.setattr(graphs, "_LOAD_ATTEMPTED", True)
        reset_embedding_state()
        assert graphs._EMBEDDING_MODEL is None
        assert graphs._EMBEDDING_BACKEND is None
        assert graphs._LOAD_ATTEMPTED is False

    def test_resets_model_name_and_version(self, monkeypatch):
        monkeypatch.setattr(graphs, "_EMBEDDING_MODEL_NAME", "custom-model")
        monkeypatch.setattr(graphs, "_EMBEDDING_MODEL_VERSION", "v99")
        reset_embedding_state()
        assert graphs._EMBEDDING_MODEL_NAME == "bge-small-en-v1.5"
        assert graphs._EMBEDDING_MODEL_VERSION == "v1.5"

    def test_resets_onnx_model_dir(self, monkeypatch):
        monkeypatch.setattr(graphs, "_ONNX_MODEL_DIR", "/some/path")
        reset_embedding_state()
        assert graphs._ONNX_MODEL_DIR is None

    def test_resets_circuit_breaker(self, monkeypatch):
        monkeypatch.setattr(graphs, "_FIRST_FAILURE_TIME", 12345.0)
        from omega.graphs import _get_embedding_model
        _get_embedding_model._attempt_count = 3
        reset_embedding_state()
        assert graphs._FIRST_FAILURE_TIME == 0.0
        assert _get_embedding_model._attempt_count == 0


# ---------------------------------------------------------------------------
# 3. _maybe_unload_model
# ---------------------------------------------------------------------------

class TestMaybeUnloadModel:
    """Test idle-timeout unloading and rate-limiting logic."""

    def test_noop_when_no_model_loaded(self, monkeypatch):
        """Should return immediately if no model is loaded."""
        monkeypatch.setattr(graphs, "_EMBEDDING_MODEL", None)
        monkeypatch.setattr(graphs, "_LAST_UNLOAD_CHECK", 0.0)
        _maybe_unload_model()
        # No crash, no state change
        assert graphs._EMBEDDING_MODEL is None

    def test_rate_limited_check(self, monkeypatch):
        """Should skip check if called within the rate-limit interval."""
        now = time.monotonic()
        monkeypatch.setattr(graphs, "_EMBEDDING_MODEL", "fake-model")
        monkeypatch.setattr(graphs, "_LAST_EMBED_TIME", now - 9999)
        # Set last check to very recent — within interval
        monkeypatch.setattr(graphs, "_LAST_UNLOAD_CHECK", now - 10)
        _maybe_unload_model()
        # Model should NOT be unloaded because rate limit blocks the check
        assert graphs._EMBEDDING_MODEL == "fake-model"

    def test_no_unload_when_recently_active(self, monkeypatch):
        """Should not unload if last embed was recent (within idle timeout)."""
        now = time.monotonic()
        monkeypatch.setattr(graphs, "_EMBEDDING_MODEL", "fake-model")
        monkeypatch.setattr(graphs, "_LAST_EMBED_TIME", now - 60)  # 1 min ago
        monkeypatch.setattr(graphs, "_LAST_UNLOAD_CHECK", 0.0)  # force check
        _maybe_unload_model()
        assert graphs._EMBEDDING_MODEL == "fake-model"

    def test_unloads_after_idle_timeout(self, monkeypatch):
        """Should unload model when idle exceeds timeout."""
        now = time.monotonic()
        monkeypatch.setattr(graphs, "_EMBEDDING_MODEL", "fake-model")
        monkeypatch.setattr(graphs, "_EMBEDDING_BACKEND", "onnx")
        monkeypatch.setattr(graphs, "_LAST_EMBED_TIME", now - 700)  # > 600s idle
        monkeypatch.setattr(graphs, "_LAST_UNLOAD_CHECK", 0.0)  # force check
        graphs._EMBEDDING_CACHE["key1"] = [0.1, 0.2]
        _maybe_unload_model()
        assert graphs._EMBEDDING_MODEL is None
        assert graphs._EMBEDDING_BACKEND is None
        assert graphs._LAST_EMBED_TIME == 0.0
        assert len(graphs._EMBEDDING_CACHE) == 0

    def test_skips_when_last_embed_time_zero(self, monkeypatch):
        """Should not unload if _LAST_EMBED_TIME is 0.0 (never used)."""
        monkeypatch.setattr(graphs, "_EMBEDDING_MODEL", "fake-model")
        monkeypatch.setattr(graphs, "_LAST_EMBED_TIME", 0.0)
        monkeypatch.setattr(graphs, "_LAST_UNLOAD_CHECK", 0.0)
        _maybe_unload_model()
        assert graphs._EMBEDDING_MODEL == "fake-model"


# ---------------------------------------------------------------------------
# 4. _check_onnx_runtime — caching
# ---------------------------------------------------------------------------

class TestCheckOnnxRuntime:
    """Test ONNX runtime availability check with caching."""

    def test_caches_result(self, monkeypatch):
        monkeypatch.setattr(graphs, "_ONNX_CHECKED", False)
        monkeypatch.setattr(graphs, "_ONNX_AVAILABLE", False)
        result1 = _check_onnx_runtime()
        assert graphs._ONNX_CHECKED is True
        # Second call should use cached value, not re-check
        monkeypatch.setattr(graphs, "_ONNX_AVAILABLE", not result1)
        result2 = _check_onnx_runtime()
        # Returns the cached (flipped) value since _ONNX_CHECKED is still True
        assert result2 == (not result1)

    def test_returns_bool(self, monkeypatch):
        monkeypatch.setattr(graphs, "_ONNX_CHECKED", False)
        result = _check_onnx_runtime()
        assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# 5. _check_sentence_transformers — caching
# ---------------------------------------------------------------------------

class TestCheckSentenceTransformers:
    """Test sentence_transformers availability check with caching."""

    def test_caches_result(self, monkeypatch):
        monkeypatch.setattr(graphs, "_SENTENCE_TRANSFORMERS_CHECKED", False)
        monkeypatch.setattr(graphs, "_SENTENCE_TRANSFORMERS_AVAILABLE", False)
        result1 = _check_sentence_transformers()
        assert graphs._SENTENCE_TRANSFORMERS_CHECKED is True
        # Modify cached value and verify second call uses it
        monkeypatch.setattr(graphs, "_SENTENCE_TRANSFORMERS_AVAILABLE", not result1)
        result2 = _check_sentence_transformers()
        assert result2 == (not result1)

    def test_returns_bool(self, monkeypatch):
        monkeypatch.setattr(graphs, "_SENTENCE_TRANSFORMERS_CHECKED", False)
        result = _check_sentence_transformers()
        assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# 6. _hash_embedding
# ---------------------------------------------------------------------------

class TestHashEmbedding:
    """Test deterministic hash-based fallback embedding."""

    def test_correct_dimension(self):
        emb = _hash_embedding("hello world")
        assert len(emb) == 384

    def test_custom_dimension(self):
        emb = _hash_embedding("hello world", dimension=128)
        assert len(emb) == 128

    def test_normalized(self):
        emb = _hash_embedding("hello world")
        magnitude = math.sqrt(sum(x * x for x in emb))
        assert abs(magnitude - 1.0) < 1e-6

    def test_deterministic(self):
        emb1 = _hash_embedding("test input")
        emb2 = _hash_embedding("test input")
        assert emb1 == emb2

    def test_different_inputs_different_outputs(self):
        emb1 = _hash_embedding("alpha")
        emb2 = _hash_embedding("beta")
        assert emb1 != emb2

    def test_empty_string(self):
        emb = _hash_embedding("")
        assert len(emb) == 384
        magnitude = math.sqrt(sum(x * x for x in emb))
        assert abs(magnitude - 1.0) < 1e-6

    def test_very_long_string(self):
        long_text = "x" * 100_000
        emb = _hash_embedding(long_text)
        assert len(emb) == 384
        magnitude = math.sqrt(sum(x * x for x in emb))
        assert abs(magnitude - 1.0) < 1e-6

    def test_all_floats(self):
        emb = _hash_embedding("check types")
        assert all(isinstance(v, float) for v in emb)


# ---------------------------------------------------------------------------
# 7. generate_embedding
# ---------------------------------------------------------------------------

class TestGenerateEmbedding:
    """Test generate_embedding with hash fallback (no real model)."""

    def test_hash_fallback_when_no_backend(self, monkeypatch):
        """With skip flag set, should fall back to hash embedding."""
        monkeypatch.setenv("OMEGA_SKIP_EMBEDDINGS", "1")
        monkeypatch.setattr(graphs, "_ONNX_CHECKED", True)
        monkeypatch.setattr(graphs, "_ONNX_AVAILABLE", False)
        monkeypatch.setattr(graphs, "_SENTENCE_TRANSFORMERS_CHECKED", True)
        monkeypatch.setattr(graphs, "_SENTENCE_TRANSFORMERS_AVAILABLE", False)
        emb = generate_embedding("test text")
        assert len(emb) == 384
        # Should match the hash fallback output
        expected = _hash_embedding("test text")
        assert emb == expected

    def test_returns_list_of_floats(self, monkeypatch):
        monkeypatch.setenv("OMEGA_SKIP_EMBEDDINGS", "1")
        monkeypatch.setattr(graphs, "_ONNX_CHECKED", True)
        monkeypatch.setattr(graphs, "_ONNX_AVAILABLE", False)
        monkeypatch.setattr(graphs, "_SENTENCE_TRANSFORMERS_CHECKED", True)
        monkeypatch.setattr(graphs, "_SENTENCE_TRANSFORMERS_AVAILABLE", False)
        emb = generate_embedding("hello")
        assert isinstance(emb, list)
        assert all(isinstance(v, float) for v in emb)

    def test_cache_hit(self, monkeypatch):
        """Cache should return identical results for repeated calls."""
        monkeypatch.setenv("OMEGA_SKIP_EMBEDDINGS", "1")
        monkeypatch.setattr(graphs, "_ONNX_CHECKED", True)
        monkeypatch.setattr(graphs, "_ONNX_AVAILABLE", False)
        monkeypatch.setattr(graphs, "_SENTENCE_TRANSFORMERS_CHECKED", True)
        monkeypatch.setattr(graphs, "_SENTENCE_TRANSFORMERS_AVAILABLE", False)
        emb1 = generate_embedding("cached text")
        emb2 = generate_embedding("cached text")
        assert emb1 == emb2

    def test_empty_string_input(self, monkeypatch):
        monkeypatch.setenv("OMEGA_SKIP_EMBEDDINGS", "1")
        monkeypatch.setattr(graphs, "_ONNX_CHECKED", True)
        monkeypatch.setattr(graphs, "_ONNX_AVAILABLE", False)
        monkeypatch.setattr(graphs, "_SENTENCE_TRANSFORMERS_CHECKED", True)
        monkeypatch.setattr(graphs, "_SENTENCE_TRANSFORMERS_AVAILABLE", False)
        emb = generate_embedding("")
        assert len(emb) == 384


# ---------------------------------------------------------------------------
# 8. generate_embeddings_batch
# ---------------------------------------------------------------------------

class TestGenerateEmbeddingsBatch:
    """Test batch embedding generation."""

    def test_empty_input(self):
        result = generate_embeddings_batch([])
        assert result == []

    def test_hash_fallback_batch(self, monkeypatch):
        """With no backend, batch should produce hash embeddings for each text."""
        monkeypatch.setenv("OMEGA_SKIP_EMBEDDINGS", "1")
        monkeypatch.setattr(graphs, "_ONNX_CHECKED", True)
        monkeypatch.setattr(graphs, "_ONNX_AVAILABLE", False)
        monkeypatch.setattr(graphs, "_SENTENCE_TRANSFORMERS_CHECKED", True)
        monkeypatch.setattr(graphs, "_SENTENCE_TRANSFORMERS_AVAILABLE", False)
        texts = ["hello", "world", "test"]
        results = generate_embeddings_batch(texts)
        assert len(results) == 3
        for emb in results:
            assert len(emb) == 384
        # Each text should produce a different embedding
        assert results[0] != results[1]
        assert results[1] != results[2]

    def test_batch_matches_individual_hash(self, monkeypatch):
        """Batch hash fallback should match individual hash fallback outputs."""
        monkeypatch.setenv("OMEGA_SKIP_EMBEDDINGS", "1")
        monkeypatch.setattr(graphs, "_ONNX_CHECKED", True)
        monkeypatch.setattr(graphs, "_ONNX_AVAILABLE", False)
        monkeypatch.setattr(graphs, "_SENTENCE_TRANSFORMERS_CHECKED", True)
        monkeypatch.setattr(graphs, "_SENTENCE_TRANSFORMERS_AVAILABLE", False)
        texts = ["alpha", "beta"]
        batch_results = generate_embeddings_batch(texts)
        individual_results = [_hash_embedding(t) for t in texts]
        assert batch_results == individual_results

    def test_single_item_batch(self, monkeypatch):
        monkeypatch.setenv("OMEGA_SKIP_EMBEDDINGS", "1")
        monkeypatch.setattr(graphs, "_ONNX_CHECKED", True)
        monkeypatch.setattr(graphs, "_ONNX_AVAILABLE", False)
        monkeypatch.setattr(graphs, "_SENTENCE_TRANSFORMERS_CHECKED", True)
        monkeypatch.setattr(graphs, "_SENTENCE_TRANSFORMERS_AVAILABLE", False)
        results = generate_embeddings_batch(["only one"])
        assert len(results) == 1
        assert len(results[0]) == 384


# ---------------------------------------------------------------------------
# 9. _get_onnx_model_dir
# ---------------------------------------------------------------------------

class TestGetOnnxModelDir:
    """Test ONNX model directory resolution logic."""

    def test_primary_path_exists(self, tmp_path, monkeypatch):
        """If primary model dir has model.onnx, use it."""
        model_dir = tmp_path / "bge-small-en-v1.5-onnx"
        model_dir.mkdir()
        (model_dir / "model.onnx").write_text("fake")
        monkeypatch.setattr(graphs, "_ONNX_MODEL_DIR", None)
        monkeypatch.setattr(graphs, "_ONNX_DEFAULT_DIR", str(model_dir))
        result = _get_onnx_model_dir()
        assert result == str(model_dir)

    def test_env_override(self, tmp_path, monkeypatch):
        """If primary doesn't exist but env var points to valid dir, use that."""
        env_dir = tmp_path / "custom-model"
        env_dir.mkdir()
        (env_dir / "model.onnx").write_text("fake")
        # Make primary non-existent
        monkeypatch.setattr(graphs, "_ONNX_MODEL_DIR", None)
        monkeypatch.setattr(graphs, "_ONNX_DEFAULT_DIR", str(tmp_path / "nonexistent"))
        monkeypatch.setenv("OMEGA_ONNX_MODEL_DIR", str(env_dir))
        result = _get_onnx_model_dir()
        assert result == str(env_dir)

    def test_fallback_path(self, tmp_path, monkeypatch):
        """If primary and env don't exist, fall back to all-MiniLM-L6-v2."""
        fallback_dir = tmp_path / "all-MiniLM-L6-v2-onnx"
        fallback_dir.mkdir()
        (fallback_dir / "model.onnx").write_text("fake")
        monkeypatch.setattr(graphs, "_ONNX_MODEL_DIR", None)
        monkeypatch.setattr(graphs, "_ONNX_DEFAULT_DIR", str(tmp_path / "nonexistent"))
        monkeypatch.setattr(graphs, "_ONNX_FALLBACK_DIR", str(fallback_dir))
        monkeypatch.delenv("OMEGA_ONNX_MODEL_DIR", raising=False)
        result = _get_onnx_model_dir()
        assert result == str(fallback_dir)
        # Should update model name to fallback
        assert graphs._EMBEDDING_MODEL_NAME == "all-MiniLM-L6-v2"
        assert graphs._EMBEDDING_MODEL_VERSION == "v2"

    def test_no_model_available(self, tmp_path, monkeypatch):
        """If no model dir exists anywhere, return None."""
        monkeypatch.setattr(graphs, "_ONNX_MODEL_DIR", None)
        monkeypatch.setattr(graphs, "_ONNX_DEFAULT_DIR", str(tmp_path / "nope1"))
        monkeypatch.setattr(graphs, "_ONNX_FALLBACK_DIR", str(tmp_path / "nope2"))
        monkeypatch.delenv("OMEGA_ONNX_MODEL_DIR", raising=False)
        result = _get_onnx_model_dir()
        assert result is None

    def test_cached_result(self, monkeypatch):
        """If _ONNX_MODEL_DIR is already set, return it immediately."""
        monkeypatch.setattr(graphs, "_ONNX_MODEL_DIR", "/cached/path")
        result = _get_onnx_model_dir()
        assert result == "/cached/path"


# ---------------------------------------------------------------------------
# 10. preload_embedding_model
# ---------------------------------------------------------------------------

class TestPreloadEmbeddingModel:
    """Test preload_embedding_model behavior."""

    def test_returns_false_when_no_backend(self, monkeypatch):
        """If neither ONNX nor sentence-transformers available, return False."""
        monkeypatch.setattr(graphs, "_ONNX_CHECKED", True)
        monkeypatch.setattr(graphs, "_ONNX_AVAILABLE", False)
        monkeypatch.setattr(graphs, "_SENTENCE_TRANSFORMERS_CHECKED", True)
        monkeypatch.setattr(graphs, "_SENTENCE_TRANSFORMERS_AVAILABLE", False)
        result = preload_embedding_model()
        assert result is False

    def test_returns_false_when_skip_set(self, monkeypatch):
        """With OMEGA_SKIP_EMBEDDINGS=1, model load returns None, so preload returns False."""
        monkeypatch.setenv("OMEGA_SKIP_EMBEDDINGS", "1")
        # Pretend ONNX is available so it enters _get_embedding_model
        monkeypatch.setattr(graphs, "_ONNX_CHECKED", True)
        monkeypatch.setattr(graphs, "_ONNX_AVAILABLE", True)
        result = preload_embedding_model()
        assert result is False


# ---------------------------------------------------------------------------
# Auxiliary / edge case tests
# ---------------------------------------------------------------------------

class TestAuxiliary:
    """Additional coverage for helper functions and edge cases."""

    def test_get_active_backend_default(self):
        assert get_active_backend() is None

    def test_has_embedding_backend_no_model(self, monkeypatch):
        """With everything disabled, _has_embedding_backend returns False."""
        monkeypatch.setattr(graphs, "_EMBEDDING_MODEL", None)
        monkeypatch.setattr(graphs, "_ONNX_CHECKED", True)
        monkeypatch.setattr(graphs, "_ONNX_AVAILABLE", False)
        monkeypatch.setattr(graphs, "_SENTENCE_TRANSFORMERS_CHECKED", True)
        monkeypatch.setattr(graphs, "_SENTENCE_TRANSFORMERS_AVAILABLE", False)
        assert _has_embedding_backend() is False

    def test_hash_embedding_dimension_one(self):
        """Edge case: single-dimension embedding."""
        emb = _hash_embedding("test", dimension=1)
        assert len(emb) == 1
        # Single-dimension normalized vector should be +/- 1.0
        assert abs(abs(emb[0]) - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# 11. get_embedding_info
# ---------------------------------------------------------------------------

class TestGetEmbeddingInfo:
    """Test get_embedding_info() returns correct structure and values."""

    def test_returns_expected_keys(self):
        from omega.graphs import get_embedding_info
        info = get_embedding_info()
        expected_keys = {
            "backend", "model", "model_loaded", "onnx_available",
            "onnx_model_dir", "sentence_transformers_available",
            "dimension", "cache_size", "lazy_loading",
        }
        assert set(info.keys()) == expected_keys

    def test_dimension_is_384(self):
        from omega.graphs import get_embedding_info
        info = get_embedding_info()
        assert info["dimension"] == 384

    def test_lazy_loading_true(self):
        from omega.graphs import get_embedding_info
        info = get_embedding_info()
        assert info["lazy_loading"] is True

    def test_reflects_cache_size(self, monkeypatch):
        from omega.graphs import get_embedding_info
        monkeypatch.setattr(graphs, "_EMBEDDING_CACHE", {"a": [0.1], "b": [0.2]})
        info = get_embedding_info()
        assert info["cache_size"] == 2


# ---------------------------------------------------------------------------
# 12. reset_embedding_state clears cache
# ---------------------------------------------------------------------------

class TestResetClearsCache:
    """Test that reset_embedding_state clears the embedding cache."""

    def test_cache_cleared_on_reset(self):
        graphs._EMBEDDING_CACHE["stale_key"] = [0.1, 0.2, 0.3]
        assert len(graphs._EMBEDDING_CACHE) > 0
        reset_embedding_state()
        assert len(graphs._EMBEDDING_CACHE) == 0


# ---------------------------------------------------------------------------
# 13. __all__ export list
# ---------------------------------------------------------------------------

class TestAllExports:
    """Test that __all__ is defined and contains expected public API."""

    def test_all_defined(self):
        assert hasattr(graphs, "__all__")

    def test_all_contains_public_api(self):
        expected = {
            "generate_embedding",
            "generate_embeddings_batch",
            "generate_embedding_async",
            "generate_embeddings_batch_async",
            "preload_embedding_model",
            "preload_embedding_model_async",
            "get_embedding_model_info",
            "get_embedding_info",
            "get_active_backend",
            "has_onnx_runtime",
            "has_sentence_transformers",
            "reset_embedding_state",
        }
        assert set(graphs.__all__) == expected

    def test_all_entries_are_importable(self):
        for name in graphs.__all__:
            assert hasattr(graphs, name), f"{name} in __all__ but not defined"
