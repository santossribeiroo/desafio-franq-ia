"""
Unit tests for src/tools.py — database interaction and safety guardrails.
All tests run against the real SQLite file, so no mocking is needed.
"""
import json

import pytest

# Reset module-level caches before each test so tests are independent
@pytest.fixture(autouse=True)
def _reset_tools_cache():
    from src import tools
    tools._schema_store.clear()
    tools._query_cache.clear()
    yield
    tools._schema_store.clear()
    tools._query_cache.clear()


# ── get_database_schema ────────────────────────────────────────────────────────

class TestGetDatabaseSchema:
    def test_returns_non_empty_string(self):
        from src.tools import get_database_schema
        result = get_database_schema.invoke({})
        assert isinstance(result, str)
        assert len(result) > 50

    def test_contains_expected_tables(self):
        from src.tools import get_database_schema
        result = get_database_schema.invoke({})
        for table in ("clientes", "compras", "suporte", "campanhas_marketing"):
            assert table in result.lower(), f"Table '{table}' not found in schema"

    def test_result_is_cached_on_second_call(self):
        from src import tools
        from src.tools import get_database_schema

        first = get_database_schema.invoke({})
        # After the first call, _schema_store must have exactly 1 entry
        assert len(tools._schema_store) == 1

        second = get_database_schema.invoke({})
        # Second call must return identical object (same cache entry)
        assert first == second
        assert len(tools._schema_store) == 1  # still only 1 entry


# ── execute_sql_query ─────────────────────────────────────────────────────────

class TestExecuteSqlQuery:
    def test_valid_select_returns_json(self):
        from src.tools import execute_sql_query
        result = execute_sql_query.invoke({"sql": "SELECT COUNT(*) AS total FROM clientes"})
        rows = json.loads(result)
        assert isinstance(rows, list)
        assert len(rows) == 1
        assert "total" in rows[0]
        assert rows[0]["total"] > 0

    def test_non_select_is_blocked(self):
        from src.tools import execute_sql_query
        for dangerous in (
            "INSERT INTO clientes (nome) VALUES ('x')",
            "UPDATE clientes SET nome='x' WHERE id=1",
            "DELETE FROM clientes WHERE id=1",
            "DROP TABLE clientes",
        ):
            result = execute_sql_query.invoke({"sql": dangerous})
            assert result.startswith("ERROR"), f"Expected ERROR for: {dangerous}"

    def test_limit_injected_when_missing(self):
        from src import tools
        from src.tools import execute_sql_query

        result = execute_sql_query.invoke({"sql": "SELECT * FROM clientes"})
        # A cache entry must have been created, and the result must be valid JSON
        assert len(tools._query_cache) == 1
        rows = json.loads(result)
        # LIMIT 200 was injected, so we should get at most 200 rows
        assert isinstance(rows, list)
        assert len(rows) <= 200

    def test_explicit_limit_not_doubled(self):
        from src.tools import execute_sql_query
        # Should not raise and should return results (not an error about double LIMIT)
        result = execute_sql_query.invoke({"sql": "SELECT id FROM clientes LIMIT 3"})
        rows = json.loads(result)
        assert len(rows) <= 3

    def test_query_cached_on_second_call(self):
        from src import tools
        from src.tools import execute_sql_query

        sql = "SELECT COUNT(*) AS n FROM compras"
        execute_sql_query.invoke({"sql": sql})
        assert len(tools._query_cache) == 1

        execute_sql_query.invoke({"sql": sql})
        # Cache must not grow — same key reused
        assert len(tools._query_cache) == 1

    def test_invalid_sql_returns_error(self):
        from src.tools import execute_sql_query
        result = execute_sql_query.invoke({"sql": "SELECT * FROM tabela_que_nao_existe"})
        assert result.startswith("ERROR")

    def test_join_across_tables(self):
        from src.tools import execute_sql_query
        result = execute_sql_query.invoke({
            "sql": "SELECT cl.nome, SUM(co.valor) AS total FROM clientes cl JOIN compras co ON cl.id = co.cliente_id GROUP BY cl.id LIMIT 5"
        })
        rows = json.loads(result)
        assert isinstance(rows, list)
        assert len(rows) > 0
        assert "nome" in rows[0]
        assert "total" in rows[0]
