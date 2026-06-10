import sqlite3
from pathlib import Path
from typing import Any

DB_PATH = Path(__file__).parent.parent / "data" / "anexo_desafio_1.db"

def get_db_connection(db_path: Path = DB_PATH) -> sqlite3.Connection:
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found at: {db_path.resolve()}")
    
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row  # Allows accessing columns by name
    return conn

def get_schema(db_path: Path = DB_PATH) -> dict[str, list[dict[str, Any]]]:
    schema: dict[str, list[dict[str, Any]]] = {}
    
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        
        # Fetch all user-defined tables, ignoring internal sqlite tables
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type = ? AND name NOT LIKE 'sqlite_%'",
            ("table",),
        )
        tables = [row["name"] for row in cursor.fetchall()]
        
        for table in tables:
            cursor.execute(f"PRAGMA table_info({table})")
            schema[table] = [dict(row) for row in cursor.fetchall()]
            
    return schema

def format_schema_for_prompt(schema: dict[str, list[dict[str, Any]]]) -> str:
    formatted_lines = []
    for table_name, columns in schema.items():
        formatted_lines.append(f"Table: {table_name}")
        for col in columns:
            pk_suffix = " [PK]" if col["pk"] else ""
            formatted_lines.append(f"  - {col['name']} ({col['type']}){pk_suffix}")
        formatted_lines.append("")  # Empty line between tables
    return "\n".join(formatted_lines).strip()

if __name__ == "__main__":
    # Quick sanity check when running the script directly
    print(f"Connecting to: {DB_PATH.resolve()}\n")
    raw_schema = get_schema()
    
    print("=== Prompt-Ready Schema ===")
    print(format_schema_for_prompt(raw_schema))