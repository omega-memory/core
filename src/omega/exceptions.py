"""OMEGA Exception Hierarchy.

Provides structured exception types to replace bare ``except Exception``
handlers with specific catches that preserve diagnostic context.
"""


class OmegaError(Exception):
    """Base class for all OMEGA errors."""


class StorageError(OmegaError):
    """Raised when a storage operation (store, query, delete) fails."""


class EmbeddingError(OmegaError):
    """Raised when embedding generation or vector search fails."""


class CoordinationError(OmegaError):
    """Raised when multi-agent coordination operations fail."""


class CloudSyncError(OmegaError):
    """Raised when cloud sync operations fail."""


class HookError(OmegaError):
    """Raised when a hook handler encounters an error."""


class ValidationError(OmegaError):
    """Raised when input validation fails (session_id, entity_id, etc.)."""
