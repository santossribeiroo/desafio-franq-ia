import hashlib
import json
import sqlite3
import time

from langchain_core.tools import tool

from src.db_utils import DB_PATH, format_schema_for_prompt, get_schema

# ── Module-level state ────────────────────────────────────────────────────────

# Schema never changes at runtime — one DB read is enough for the whole process.
# List used as a mutable container so we avoid the `global` statement.
_schema_store: list[str] = []

# SQL result cache: avoids re-running identical queries within the same session.
# Keyed by MD5 of the normalised SQL string; LRU-style eviction when full.
_query_cache: dict[str, str] = {}
_MAX_CACHE_ENTRIES = 50

_MAX_ROWS = 200           # Hard cap to prevent accidental full-table scans
_QUERY_TIMEOUT_SEC = 10.0  # Any query running longer than this is interrupted


# ── Helpers ───────────────────────────────────────────────────────────────────

def _cache_key(sql: str) -> str:
    """Produce a stable hash from normalised SQL for cache lookups."""
    normalised = " ".join(sql.upper().split())
    return hashlib.md5(normalised.encode()).hexdigest()


def _make_timeout_handler(start: float, limit: float):
    """
    Return a SQLite progress-handler callback that interrupts a long-running
    query. SQLite calls the handler every N opcodes; returning non-zero causes
    an OperationalError('interrupted').
    """
    def handler():
        return 1 if time.monotonic() - start > limit else 0
    return handler


# ── Tools ─────────────────────────────────────────────────────────────────────

@tool
def get_database_schema() -> str:
    """
    Returns the complete schema of the SQLite database, listing all tables and
    their columns with data types. Always call this tool first before writing
    any SQL query to understand what data is available.
    """
    # Using a mutable container at module level avoids the `global` statement
    # while still giving us a single cached value for the process lifetime.
    if not _schema_store:
        _schema_store.append(format_schema_for_prompt(get_schema()))
    return _schema_store[0]


@tool
def execute_sql_query(sql: str) -> str:
    """
    Executes a read-only SELECT query against the database and returns results
    as a JSON string. You can call this tool multiple times.

    Args:
        sql: A valid SQLite SELECT statement. Only SELECT is allowed.
    """
    first_token = sql.strip().split()[0].upper() if sql.strip() else ""
    if first_token != "SELECT":
        return "ERROR: Only SELECT statements are permitted."

    # Inject LIMIT if the query has none — prevents expensive full-table scans
    if "LIMIT" not in sql.upper():
        sql = f"{sql.rstrip('; ')} LIMIT {_MAX_ROWS}"

    key = _cache_key(sql)
    if key in _query_cache:
        # Identical query already executed this session — return cached result
        return _query_cache[key]

    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row

    try:
        conn.execute("PRAGMA query_only = ON")
        conn.set_progress_handler(_make_timeout_handler(time.monotonic(), _QUERY_TIMEOUT_SEC), 100)

        cursor = conn.cursor()
        cursor.execute(f"EXPLAIN QUERY PLAN {sql}")  # syntax validation before execution
        cursor.execute(sql)
        rows = [dict(row) for row in cursor.fetchall()]

        if not rows:
            return "Query executed successfully but returned no results. Consider adjusting filters or date ranges."

        result = json.dumps(rows, ensure_ascii=False, indent=2)

        # LRU-style eviction: drop oldest entry when cache is full
        if len(_query_cache) >= _MAX_CACHE_ENTRIES:
            del _query_cache[next(iter(_query_cache))]
        _query_cache[key] = result

        return result

    except sqlite3.OperationalError as exc:
        if "interrupted" in str(exc).lower():
            return f"ERROR: Query exceeded {_QUERY_TIMEOUT_SEC:.0f}s timeout. Use more specific filters to narrow the result set."
        return f"ERROR: {exc}"

    except (sqlite3.Error, ValueError, OSError) as exc:
        return f"ERROR: {exc}"

    finally:
        conn.set_progress_handler(None, 0)  # always remove the handler
        conn.close()
