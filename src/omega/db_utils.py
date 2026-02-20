"""Shared SQLite utilities for OMEGA.

Centralizes retry logic and connection helpers used across sqlite_store.py
and coordination.py to avoid code duplication.
"""

import logging
import sqlite3
import time

logger = logging.getLogger("omega.db_utils")

# SQLite retry — handles multi-process write contention on shared omega.db.
# WAL mode + busy_timeout handle most cases, but under heavy contention
# (3+ MCP server processes) the busy_timeout can still expire. This wrapper
# retries with exponential backoff before surfacing the error.
DB_RETRY_ATTEMPTS = 3
DB_RETRY_BASE_DELAY = 1.0  # seconds


def retry_on_locked(fn, *args, **kwargs):
    """Call fn with retry on 'database is locked' OperationalError."""
    for attempt in range(DB_RETRY_ATTEMPTS):
        try:
            return fn(*args, **kwargs)
        except sqlite3.OperationalError as e:
            if "database is locked" in str(e) and attempt < DB_RETRY_ATTEMPTS - 1:
                delay = DB_RETRY_BASE_DELAY * (2 ** attempt)
                logger.warning("database is locked (attempt %d/%d), retrying in %.1fs",
                               attempt + 1, DB_RETRY_ATTEMPTS, delay)
                time.sleep(delay)
            else:
                raise


def retry_write_on_locked(conn, fn, *args, **kwargs):
    """Call fn with retry on 'database is locked', rolling back between attempts.

    Unlike retry_on_locked (which wraps a single commit), this wraps an entire
    write transaction that may include multiple execute() calls before commit.
    On failure, the uncommitted transaction is rolled back before retrying so
    the next attempt starts with a clean slate.
    """
    for attempt in range(DB_RETRY_ATTEMPTS):
        try:
            return fn(*args, **kwargs)
        except sqlite3.OperationalError as e:
            if "database is locked" in str(e) and attempt < DB_RETRY_ATTEMPTS - 1:
                delay = DB_RETRY_BASE_DELAY * (2 ** attempt)
                logger.warning("database is locked (attempt %d/%d), rolling back and retrying in %.1fs",
                               attempt + 1, DB_RETRY_ATTEMPTS, delay)
                try:
                    conn.rollback()
                except Exception:
                    pass
                time.sleep(delay)
            else:
                raise
