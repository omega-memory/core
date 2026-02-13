"""
OMEGA Embeddings — ONNX-based embedding generation for semantic search.

Provides:
- generate_embedding(text) → 384-dim normalized vector
- generate_embeddings_batch(texts) → list of vectors
- Async variants for non-blocking operation
- LRU cache for repeated queries
- Hash-based fallback when no ML model available

Uses bge-small-en-v1.5 via ONNX Runtime (~90MB RAM, ~170MB with arena disabled).
Falls back to all-MiniLM-L6-v2 if bge model not downloaded, or SentenceTransformers (PyTorch) if ONNX unavailable.
"""

from collections import OrderedDict
from typing import Dict, List, Optional, Any
import hashlib
import logging
import math

__all__ = [
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
]

logger = logging.getLogger("omega.graphs")

# NumPy for vectorized operations (required by ONNX backend)
try:
    import numpy as np
except ImportError:
    np = None  # type: ignore[assignment]  # ONNX path won't be entered

# Embedding model state
_EMBEDDING_MODEL = None
_EMBEDDING_BACKEND = None  # "onnx" or "sentence-transformers"
_LOAD_ATTEMPTED = False  # Circuit breaker: don't retry after first failure
_EMBEDDING_MODEL_NAME = "bge-small-en-v1.5"
_EMBEDDING_MODEL_VERSION = "v1.5"
_EMBEDDING_CACHE: OrderedDict = OrderedDict()
_EMBEDDING_CACHE_MAX = 512

# Idle-timeout model unloading — save RAM when not actively embedding
import time as _time_module

_LAST_EMBED_TIME: float = 0.0  # monotonic timestamp of last embedding call
_LAST_UNLOAD_CHECK: float = 0.0  # monotonic timestamp of last idle check
_IDLE_TIMEOUT_S = 600  # 10 minutes
_UNLOAD_CHECK_INTERVAL_S = 60  # check at most once per 60s

# Circuit breaker cooldown — allow retry after transient failures
_FIRST_FAILURE_TIME: float = 0.0  # monotonic timestamp of first failure in current window
_CIRCUIT_BREAKER_COOLDOWN_S = 300  # 5 minutes

# ONNX model paths (bge-small-en-v1.5 primary, all-MiniLM-L6-v2 fallback)
_ONNX_MODEL_DIR = None
_ONNX_DEFAULT_DIR = "~/.cache/omega/models/bge-small-en-v1.5-onnx"
_ONNX_FALLBACK_DIR = "~/.cache/omega/models/all-MiniLM-L6-v2-onnx"

# Availability checks (cached)
_ONNX_CHECKED = False
_ONNX_AVAILABLE = False
_SENTENCE_TRANSFORMERS_CHECKED = False
_SENTENCE_TRANSFORMERS_AVAILABLE = False


def get_embedding_model_info() -> Dict[str, Any]:
    """Return current embedding model name and version for metadata tracking."""
    return {
        "model_name": _EMBEDDING_MODEL_NAME,
        "model_version": _EMBEDDING_MODEL_VERSION,
        "model_loaded": _EMBEDDING_MODEL is not None,
        "backend": _EMBEDDING_BACKEND,
    }


def reset_embedding_state():
    """Reset all embedding state including the circuit breaker.

    Call this after tests that set OMEGA_SKIP_EMBEDDINGS=1 to allow
    subsequent tests to load the real model.
    """
    global _EMBEDDING_MODEL, _EMBEDDING_BACKEND, _LOAD_ATTEMPTED
    global _ONNX_MODEL_DIR, _EMBEDDING_MODEL_NAME, _EMBEDDING_MODEL_VERSION
    global _FIRST_FAILURE_TIME
    _EMBEDDING_MODEL = None
    _EMBEDDING_BACKEND = None
    _LOAD_ATTEMPTED = False
    _ONNX_MODEL_DIR = None
    _EMBEDDING_MODEL_NAME = "bge-small-en-v1.5"
    _EMBEDDING_MODEL_VERSION = "v1.5"
    _FIRST_FAILURE_TIME = 0.0
    _EMBEDDING_CACHE.clear()
    if hasattr(_get_embedding_model, "_attempt_count"):
        _get_embedding_model._attempt_count = 0


def _maybe_unload_model():
    """Unload the embedding model if idle > 10 min to save RAM.

    Rate-limited to check at most once per 60s. On unload, resets the
    attempt count so the model can be lazy-reloaded on next call.
    Always unload when idle — don't gate on RSS threshold.
    """
    global _EMBEDDING_MODEL, _EMBEDDING_BACKEND, _LAST_UNLOAD_CHECK, _LAST_EMBED_TIME, _FIRST_FAILURE_TIME
    now = _time_module.monotonic()
    if now - _LAST_UNLOAD_CHECK < _UNLOAD_CHECK_INTERVAL_S:
        return
    _LAST_UNLOAD_CHECK = now

    if _EMBEDDING_MODEL is None:
        return
    if _LAST_EMBED_TIME == 0.0:
        return
    if now - _LAST_EMBED_TIME < _IDLE_TIMEOUT_S:
        return

    logger.info("Idle-unloading embedding model (idle > %ds)", _IDLE_TIMEOUT_S)
    _EMBEDDING_MODEL = None
    _EMBEDDING_BACKEND = None
    _LAST_EMBED_TIME = 0.0  # Reset so idle detection works correctly on reload
    _EMBEDDING_CACHE.clear()  # Free cached embeddings too
    # Reset circuit breaker so model can reload
    _FIRST_FAILURE_TIME = 0.0
    if hasattr(_get_embedding_model, "_attempt_count"):
        _get_embedding_model._attempt_count = 0


def _check_onnx_runtime() -> bool:
    """Check if onnxruntime is available (lazy check, cached result)."""
    global _ONNX_CHECKED, _ONNX_AVAILABLE
    if not _ONNX_CHECKED:
        try:
            import importlib.util

            _ONNX_AVAILABLE = importlib.util.find_spec("onnxruntime") is not None
        except Exception:
            _ONNX_AVAILABLE = False
        _ONNX_CHECKED = True
    return _ONNX_AVAILABLE


def _check_sentence_transformers() -> bool:
    """Check if sentence_transformers is available (lazy check, cached result)."""
    global _SENTENCE_TRANSFORMERS_CHECKED, _SENTENCE_TRANSFORMERS_AVAILABLE
    if not _SENTENCE_TRANSFORMERS_CHECKED:
        try:
            import importlib.util

            _SENTENCE_TRANSFORMERS_AVAILABLE = importlib.util.find_spec("sentence_transformers") is not None
        except Exception:
            _SENTENCE_TRANSFORMERS_AVAILABLE = False
        _SENTENCE_TRANSFORMERS_CHECKED = True
    return _SENTENCE_TRANSFORMERS_AVAILABLE


def has_sentence_transformers() -> bool:
    """Check if sentence_transformers is available."""
    return _check_sentence_transformers()


def has_onnx_runtime() -> bool:
    """Check if onnxruntime is available."""
    return _check_onnx_runtime()


def _get_onnx_model_dir() -> Optional[str]:
    """Get the ONNX model directory path, checking if model exists.

    Checks in order: bge-small-en-v1.5 (primary), env override, all-MiniLM-L6-v2 (fallback).
    """
    global _ONNX_MODEL_DIR, _EMBEDDING_MODEL_NAME, _EMBEDDING_MODEL_VERSION
    if _ONNX_MODEL_DIR is not None:
        return _ONNX_MODEL_DIR

    import os
    from pathlib import Path

    # Primary: bge-small-en-v1.5
    model_dir = Path(os.path.expanduser(_ONNX_DEFAULT_DIR))
    model_path = model_dir / "model.onnx"
    if model_path.exists():
        _ONNX_MODEL_DIR = str(model_dir)
        return _ONNX_MODEL_DIR

    # Environment override
    env_dir = os.environ.get("OMEGA_ONNX_MODEL_DIR")
    if env_dir:
        env_path = Path(env_dir) / "model.onnx"
        if env_path.exists():
            _ONNX_MODEL_DIR = env_dir
            return _ONNX_MODEL_DIR

    # Fallback: all-MiniLM-L6-v2 (existing installs)
    fallback_dir = Path(os.path.expanduser(_ONNX_FALLBACK_DIR))
    fallback_path = fallback_dir / "model.onnx"
    if fallback_path.exists():
        _ONNX_MODEL_DIR = str(fallback_dir)
        _EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
        _EMBEDDING_MODEL_VERSION = "v2"
        logger.info("Using fallback model all-MiniLM-L6-v2. Run 'omega setup --download-model' to upgrade.")
        return _ONNX_MODEL_DIR

    return None


def _get_embedding_model():
    """Lazy-load the embedding model.

    Priority: ONNX Runtime (~90MB) > SentenceTransformer (~1GB PyTorch)
    """
    global _EMBEDDING_MODEL, _EMBEDDING_BACKEND, _LOAD_ATTEMPTED
    if _EMBEDDING_MODEL is not None:
        return _EMBEDDING_MODEL

    # Allow up to 3 load attempts before giving up; reset after cooldown
    global _FIRST_FAILURE_TIME
    if not hasattr(_get_embedding_model, "_attempt_count"):
        _get_embedding_model._attempt_count = 0
    if _get_embedding_model._attempt_count >= 3:
        # Time-based recovery: reset after cooldown period
        if _FIRST_FAILURE_TIME > 0 and (_time_module.monotonic() - _FIRST_FAILURE_TIME) >= _CIRCUIT_BREAKER_COOLDOWN_S:
            _get_embedding_model._attempt_count = 0
            _FIRST_FAILURE_TIME = 0.0
            logger.info("Circuit breaker cooldown expired, retrying model load")
        else:
            return None
    _get_embedding_model._attempt_count += 1
    if _get_embedding_model._attempt_count == 1:
        _FIRST_FAILURE_TIME = _time_module.monotonic()

    _LOAD_ATTEMPTED = True

    import os

    os.environ.setdefault("TQDM_DISABLE", "1")

    if os.environ.get("OMEGA_SKIP_EMBEDDINGS") == "1":
        logger.info("Skipping embedding model load (OMEGA_SKIP_EMBEDDINGS=1)")
        return None

    # Try ONNX Runtime first
    if _check_onnx_runtime():
        onnx_dir = _get_onnx_model_dir()
        if onnx_dir:
            try:
                import onnxruntime as ort
                from tokenizers import Tokenizer as FastTokenizer

                tokenizer = FastTokenizer.from_file(f"{onnx_dir}/tokenizer.json")
                tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")
                tokenizer.enable_truncation(max_length=512)
                sess_opts = ort.SessionOptions()
                sess_opts.log_severity_level = 4
                sess_opts.log_verbosity_level = 0
                sess_opts.enable_cpu_mem_arena = False  # Save ~50MB RAM
                # CPU-only: CoreML leaks ~700KB/op in native memory on long-running
                # processes (profiled Feb 2026). Speed difference is negligible for
                # single 384-dim embeddings (<5ms either way).
                import contextlib
                import io

                providers = ["CPUExecutionProvider"]
                with contextlib.redirect_stderr(io.StringIO()):
                    session = ort.InferenceSession(
                        f"{onnx_dir}/model.onnx",
                        sess_options=sess_opts,
                        providers=providers,
                    )
                _EMBEDDING_MODEL = (tokenizer, session)
                _EMBEDDING_BACKEND = "onnx"
                _get_embedding_model._attempt_count = 0
                _FIRST_FAILURE_TIME = 0.0
                logger.info("Loaded ONNX embedding model")
                return _EMBEDDING_MODEL
            except Exception as e:
                logger.warning(f"Failed to load ONNX model (attempt {_get_embedding_model._attempt_count}): {e}")
                import traceback

                logger.debug(f"ONNX load traceback: {traceback.format_exc()}")

    # Fallback to SentenceTransformer
    if _check_sentence_transformers():
        try:
            from sentence_transformers import SentenceTransformer

            _EMBEDDING_MODEL = SentenceTransformer("BAAI/bge-small-en-v1.5")
            _EMBEDDING_BACKEND = "sentence-transformers"
            _get_embedding_model._attempt_count = 0
            _FIRST_FAILURE_TIME = 0.0
            logger.info("Loaded sentence-transformers model (PyTorch fallback)")
        except Exception as e:
            logger.warning(f"Failed to load sentence-transformers: {e}")
            _EMBEDDING_MODEL = None

    if _EMBEDDING_MODEL is None:
        logger.warning(
            f"No embedding model loaded after attempt {_get_embedding_model._attempt_count}/3. "
            f"ONNX available: {_check_onnx_runtime()}, "
            f"ONNX dir: {_get_onnx_model_dir()}, "
            f"SentenceTransformers available: {_check_sentence_transformers()}"
        )
        # User-visible warning on first failure only (not every retry)
        if _get_embedding_model._attempt_count == 1 and _get_onnx_model_dir() is None:
            import sys
            print(
                "\n  WARNING: ONNX embedding model not found.\n"
                "  Semantic search is disabled — queries will use text matching only.\n"
                "  Run 'omega setup' to download the model (~90 MB).\n",
                file=sys.stderr,
            )

    return _EMBEDDING_MODEL


def preload_embedding_model() -> bool:
    """Preload the embedding model at startup (optional warmup)."""
    if not _check_onnx_runtime() and not _check_sentence_transformers():
        return False
    try:
        model = _get_embedding_model()
        if model is not None:
            generate_embedding("test")
            return True
    except Exception:
        pass
    return False


def get_active_backend() -> Optional[str]:
    """Return the currently active embedding backend, or None if using hash fallback."""
    return _EMBEDDING_BACKEND


def _hash_embedding(text: str, dimension: int = 384) -> List[float]:
    """Fallback: deterministic pseudo-embedding from text hash."""
    hash_digest = hashlib.md5(text.encode()).digest()
    seed = int.from_bytes(hash_digest[:4], byteorder="big")

    import random

    rng = random.Random(seed)
    vector = [rng.gauss(0, 1) for _ in range(dimension)]

    magnitude = math.sqrt(sum(x * x for x in vector))
    if magnitude == 0:
        return [1.0 / math.sqrt(dimension)] * dimension
    return [x / magnitude for x in vector]


def _has_embedding_backend() -> bool:
    """Check if any embedding backend is available."""
    return _EMBEDDING_MODEL is not None or _check_onnx_runtime() or _check_sentence_transformers()


def _onnx_encode(tokenizer, session, texts: List[str]) -> "np.ndarray":
    """Encode texts using ONNX Runtime. Returns normalized embeddings."""
    batch = tokenizer.encode_batch(texts)
    ids = np.array([b.ids for b in batch], dtype=np.int64)
    mask = np.array([b.attention_mask for b in batch], dtype=np.int64)
    feed = {"input_ids": ids, "attention_mask": mask}
    input_names = {i.name for i in session.get_inputs()}
    if "token_type_ids" in input_names:
        feed["token_type_ids"] = np.zeros_like(ids)
    outputs = session.run(None, feed)
    embeddings = outputs[1] if len(outputs) > 1 else outputs[0]
    if embeddings.ndim == 3:
        mask_expanded = mask[:, :, np.newaxis].astype(np.float32)
        sum_emb = np.sum(embeddings * mask_expanded, axis=1)
        sum_mask = np.clip(np.sum(mask_expanded, axis=1), a_min=1e-9, a_max=None)
        embeddings = sum_emb / sum_mask
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / np.clip(norms, a_min=1e-9, a_max=None)


def generate_embedding(text: str, dimension: int = 384) -> List[float]:
    """Generate semantic embedding from text. Returns 384-dim normalized vector."""
    global _LAST_EMBED_TIME
    # Check for idle-timeout unloading opportunity
    _maybe_unload_model()

    if _has_embedding_backend():
        cache_key = hashlib.md5(text.encode()).hexdigest()
        if cache_key in _EMBEDDING_CACHE:
            _EMBEDDING_CACHE.move_to_end(cache_key)
            _LAST_EMBED_TIME = _time_module.monotonic()
            return _EMBEDDING_CACHE[cache_key]

    if _has_embedding_backend():
        try:
            model = _get_embedding_model()
            if model is not None:
                if _EMBEDDING_BACKEND == "onnx":
                    tokenizer, session = model
                    emb = _onnx_encode(tokenizer, session, [text])
                    result = emb[0].tolist()
                else:
                    embedding = model.encode(text, normalize_embeddings=True)
                    result = embedding.tolist()
                _EMBEDDING_CACHE[cache_key] = result
                while len(_EMBEDDING_CACHE) > _EMBEDDING_CACHE_MAX:
                    _EMBEDDING_CACHE.popitem(last=False)
                _LAST_EMBED_TIME = _time_module.monotonic()
                return result
            else:
                logger.warning("Embedding model is None — circuit-breaker tripped. Using hash fallback.")
        except Exception as e:
            logger.warning(f"Embedding generation failed, falling back to hash: {e}")

    return _hash_embedding(text, dimension)


def generate_embeddings_batch(texts: List[str]) -> List[List[float]]:
    """Generate embeddings for multiple texts in a single batch."""
    if not texts:
        return []

    if _has_embedding_backend():
        try:
            model = _get_embedding_model()
            if model is not None:
                if _EMBEDDING_BACKEND == "onnx":
                    tokenizer, session = model
                    # CoreML can't handle multi-item batches — use batch_size=1
                    providers = session.get_providers()
                    batch_size = 1 if any("CoreML" in p for p in providers) else 32
                    all_results = []
                    for i in range(0, len(texts), batch_size):
                        batch = texts[i : i + batch_size]
                        embs = _onnx_encode(tokenizer, session, batch)
                        all_results.extend(embs.tolist())
                    return all_results
                else:
                    embeddings = model.encode(texts, normalize_embeddings=True, batch_size=32)
                    return [e.tolist() for e in embeddings]
            else:
                logger.warning(
                    "Batch embedding: model is None (load failed or circuit-breaker tripped). "
                    "Falling back to hash embeddings — these will NOT be findable by vector search."
                )
        except Exception as e:
            logger.warning(f"Batch embedding failed, falling back to single-item: {e}")
            # Fall back to per-item encoding before resorting to hash
            try:
                results = []
                for text in texts:
                    emb = generate_embedding(text)
                    results.append(emb)
                if get_active_backend() is not None:
                    return results
                logger.warning("Single-item fallback produced hash embeddings")
            except Exception as e2:
                logger.warning(f"Single-item fallback also failed: {e2}")

    logger.warning(f"Using hash-fallback embeddings for {len(texts)} texts")
    return [_hash_embedding(t) for t in texts]


def get_embedding_info() -> Dict[str, Any]:
    """Get info about the current embedding backend."""
    has_onnx = _check_onnx_runtime()
    has_st = _check_sentence_transformers()
    if _EMBEDDING_BACKEND:
        backend = _EMBEDDING_BACKEND
    elif has_onnx:
        backend = "onnx (not loaded)"
    elif has_st:
        backend = "sentence-transformers (not loaded)"
    else:
        backend = "hash-fallback"
    return {
        "backend": backend,
        "model": _EMBEDDING_MODEL_NAME,
        "model_loaded": _EMBEDDING_MODEL is not None,
        "onnx_available": has_onnx,
        "onnx_model_dir": _get_onnx_model_dir() if has_onnx else None,
        "sentence_transformers_available": has_st,
        "dimension": 384,
        "cache_size": len(_EMBEDDING_CACHE),
        "lazy_loading": True,
    }


# Async support
import asyncio
from concurrent.futures import ThreadPoolExecutor

_EMBEDDING_EXECUTOR: Optional[ThreadPoolExecutor] = None


def _get_embedding_executor() -> ThreadPoolExecutor:
    """Get or create the thread pool executor for embedding operations."""
    global _EMBEDDING_EXECUTOR
    if _EMBEDDING_EXECUTOR is None:
        _EMBEDDING_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="embedding")
    return _EMBEDDING_EXECUTOR


async def generate_embedding_async(text: str, dimension: int = 384) -> List[float]:
    """Generate embedding asynchronously (non-blocking)."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_get_embedding_executor(), generate_embedding, text, dimension)


async def generate_embeddings_batch_async(texts: List[str]) -> List[List[float]]:
    """Generate batch embeddings asynchronously."""
    if not texts:
        return []
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_get_embedding_executor(), generate_embeddings_batch, texts)


async def preload_embedding_model_async() -> bool:
    """Preload embedding model asynchronously."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_get_embedding_executor(), preload_embedding_model)
