"""
OMEGA Crypto -- Optional encryption at rest for memory stores.

When enabled, all graph JSON and JSONL entries are encrypted with Fernet
(AES-128-CBC + HMAC-SHA256). The encryption key is derived from a
machine-specific secret stored at ~/.omega/.key.

Enable: OMEGA_ENCRYPT=1
Disable: unset OMEGA_ENCRYPT (default)

The key file is created automatically on first use. Guard it with
filesystem permissions (0600). Losing it means losing access to
encrypted memories.
"""

import base64
import logging
import os
import secrets
from pathlib import Path

logger = logging.getLogger("omega.crypto")

_fernet_instance = None
_checked = False


def _omega_home() -> Path:
    """Resolve OMEGA_HOME lazily so tests can override via env var."""
    return Path(os.environ.get("OMEGA_HOME", str(Path.home() / ".omega")))


def _key_path() -> Path:
    """Resolve key file path lazily."""
    return _omega_home() / ".key"


def is_enabled() -> bool:
    """Check if encryption at rest is enabled."""
    return os.environ.get("OMEGA_ENCRYPT", "").strip() in ("1", "true", "yes")


def reset_crypto_state() -> None:
    """Reset module state for test isolation."""
    global _fernet_instance, _checked
    _fernet_instance = None
    _checked = False


def _get_or_create_key() -> bytes:
    """Get the Fernet key, creating one if it doesn't exist."""
    kp = _key_path()
    if kp.exists():
        raw = kp.read_bytes().strip()
        # If it's a 32-byte raw secret, derive Fernet key from it
        if len(raw) == 32:
            return base64.urlsafe_b64encode(raw)
        # Already a base64-encoded Fernet key
        return raw

    # Generate a new 32-byte secret and derive Fernet key
    omega_home = _omega_home()
    omega_home.mkdir(parents=True, exist_ok=True, mode=0o700)
    raw_secret = secrets.token_bytes(32)
    encoded_key = base64.urlsafe_b64encode(raw_secret)
    # Atomic creation with restricted permissions (no TOCTOU window)
    fd = os.open(str(kp), os.O_CREAT | os.O_WRONLY | os.O_EXCL, 0o600)
    try:
        os.write(fd, encoded_key)
    finally:
        os.close(fd)
    logger.info(f"Created encryption key at {kp}")
    return encoded_key


def _get_fernet():
    """Lazy-load Fernet instance."""
    global _fernet_instance, _checked
    if _fernet_instance is not None:
        return _fernet_instance
    if _checked:
        return None
    _checked = True

    try:
        from cryptography.fernet import Fernet
    except ImportError:
        logger.warning("OMEGA_ENCRYPT=1 but 'cryptography' package not installed. Run: pip install cryptography")
        return None

    try:
        key = _get_or_create_key()
        _fernet_instance = Fernet(key)
        return _fernet_instance
    except Exception as e:
        logger.error(f"Failed to initialize encryption: {e}")
        return None


def encrypt(plaintext: str) -> str:
    """Encrypt a string. Returns base64-encoded ciphertext.

    If encryption is disabled or unavailable, returns plaintext unchanged.
    """
    if not is_enabled():
        return plaintext

    f = _get_fernet()
    if f is None:
        return plaintext

    token = f.encrypt(plaintext.encode("utf-8"))
    return "ENC:" + token.decode("ascii")


def decrypt(data: str) -> str:
    """Decrypt a string. Handles both encrypted and plaintext inputs.

    If the data doesn't start with 'ENC:', it's returned as-is (plaintext).
    This allows transparent migration: old plaintext files are readable,
    new writes are encrypted.
    """
    if not data.startswith("ENC:"):
        return data

    f = _get_fernet()
    if f is None:
        logger.error("Cannot decrypt: encryption not available")
        return data

    try:
        token = data[4:].encode("ascii")
        return f.decrypt(token).decode("utf-8")
    except Exception as e:
        logger.error(f"Decryption failed: {e}")
        return data


def encrypt_line(line: str) -> str:
    """Encrypt a single JSONL line."""
    if not is_enabled():
        return line
    return encrypt(line)


def decrypt_line(line: str) -> str:
    """Decrypt a single JSONL line. Handles plaintext transparently."""
    return decrypt(line.strip())


def secure_connect(db_path, **kwargs):
    """Create a SQLite connection with secure file permissions (0o600).

    Pre-creates the DB file with restricted permissions before connecting,
    and fixes existing files that have overly permissive permissions.
    """
    import sqlite3
    import stat

    db_path_str = str(db_path)
    path_obj = Path(db_path_str)

    if not path_obj.exists():
        # Pre-create with restricted permissions (no TOCTOU window)
        fd = os.open(db_path_str, os.O_CREAT | os.O_WRONLY, 0o600)
        os.close(fd)
    else:
        # Fix existing files with overly permissive permissions
        current_mode = path_obj.stat().st_mode
        if current_mode & (stat.S_IRWXG | stat.S_IRWXO):
            os.chmod(db_path_str, 0o600)

    return sqlite3.connect(db_path_str, **kwargs)
