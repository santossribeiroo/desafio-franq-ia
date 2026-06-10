"""
Database utility module for connecting to and inspecting the SQLite database.

The schema extraction function is a critical component of the RAG pipeline:
it provides the LLM with accurate, up-to-date table/column metadata so it
can generate valid SQL without hallucinating column or table names.
"""

import sqlite3
from pathlib import Path
from typing import Any

DB_PATH = Path(__file__).parent.parent / "data" / "anexo_desafio_1.db"


def get_db_connection(db_path: Path = DB_PATH) -> sqlite3.Connection:
    """
    Opens and returns a SQLite connection with row_factory set to
    sqlite3.Row so results can be accessed both by index and by column name.
    """
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found at: {db_path.resolve()}")

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def get_schema(db_path: Path = DB_PATH) -> dict[str, list[dict[str, Any]]]:
    """
    Dynamically extracts the full schema from the SQLite database.

    Returns a dict mapping each user-defined table name to a list of column
    descriptors. Each descriptor contains 'name', 'type', 'notnull',
    'default_value', and 'is_primary_key'.

    Filtering out sqlite_* system tables ensures only application-level
    tables are exposed to the LLM prompt, reducing noise and token usage.
    """
    schema: dict[str, list[dict[str, Any]]] = {}

    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()

        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type = ? AND name NOT LIKE 'sqlite_%'",
            ("table",),
        )
        tables = [row["name"] for row in cursor.fetchall()]

        for table in tables:
            cursor.execute(f"PRAGMA table_info({table})")
            columns = [
                {
                    "name": col["name"],
                    "type": col["type"],
                    "notnull": bool(col["notnull"]),
                    "default_value": col["dflt_value"],
                    "is_primary_key": bool(col["pk"]),
                }
                for col in cursor.fetchall()
            ]
            schema[table] = columns

    return schema


def format_schema_for_prompt(schema: dict[str, list[dict[str, Any]]]) -> str:
    """
    Renders the schema dict as a clean, human-readable string suitable for
    direct injection into an LLM prompt.

    A structured textual format (rather than raw JSON) keeps the prompt concise
    and closely mirrors the DDL style the model was trained on, improving SQL
    generation accuracy.
    """
    lines: list[str] = []

    for table, columns in schema.items():
        lines.append(f"Table: {table}")
        for col in columns:
            pk_marker = " [PK]" if col["is_primary_key"] else ""
            nn_marker = " NOT NULL" if col["notnull"] else ""
            default = f" DEFAULT {col['default_value']}" if col["default_value"] is not None else ""
            lines.append(f"  - {col['name']} ({col['type']}{pk_marker}{nn_marker}{default})")
        lines.append("")

    return "\n".join(lines).strip()


if __name__ == "__main__":
    print(f"Connecting to: {DB_PATH.resolve()}\n")

    db_schema = get_schema()

    print("=== Raw Schema Dict ===")
    for tbl_name, tbl_columns in db_schema.items():
        print(f"\n[{tbl_name}]")
        for tbl_col in tbl_columns:
            print(f"  {tbl_col}")

    print("\n\n=== Prompt-Ready Schema ===\n")
    print(format_schema_for_prompt(db_schema))
