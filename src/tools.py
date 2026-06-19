import json
import sqlite3

from langchain_core.tools import tool

from src.db_utils import DB_PATH, format_schema_for_prompt, get_schema

# Module-level cache: schema never changes at runtime, so one DB read is enough
# for the entire process lifetime — avoids a redundant round-trip on every question.
_schema_cache: str | None = None


@tool
def get_database_schema() -> str:
    """
    Returns the complete schema of the SQLite database, listing all tables and their columns with data types.
    Always call this tool first before writing any SQL query to understand what data is available.
    """
    global _schema_cache
    if _schema_cache is None:
        _schema_cache = format_schema_for_prompt(get_schema())
    return _schema_cache


@tool
def execute_sql_query(sql: str) -> str:
    """
    Executes a read-only SELECT query against the database and returns the results as a JSON string.
    You can call this tool multiple times to retrieve different pieces of data.

    Args:
        sql: A valid SQLite SELECT statement. Only SELECT is allowed — no INSERT, UPDATE, DELETE, DROP, etc.
    """
    first_token = sql.strip().split()[0].upper() if sql.strip() else ""
    if first_token != "SELECT":
        return "ERROR: Only SELECT statements are permitted."

    # PRAGMA query_only enforces read-only at the SQLite engine level, cross-platform
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row

    try:
        conn.execute("PRAGMA query_only = ON")
        cursor = conn.cursor()

        # Validate query plan before execution — catches syntax errors without side effects
        cursor.execute(f"EXPLAIN QUERY PLAN {sql}")
        cursor.execute(sql)

        rows = [dict(row) for row in cursor.fetchall()]

        if not rows:
            return "Query executed successfully but returned no results. Consider adjusting filters or date ranges."

        return json.dumps(rows, ensure_ascii=False, indent=2)

    except (sqlite3.Error, ValueError, OSError) as exc:
        return f"ERROR: {exc}"

    finally:
        conn.close()
