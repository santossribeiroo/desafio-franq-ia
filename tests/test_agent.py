"""
Smoke tests for src/agent.py — verify the agent builds correctly and
the system prompt enforces the schema-first requirement.
"""
import pytest


class TestBuildAgent:
    def test_build_agent_returns_runnable(self):
        """Agent must be a LangGraph CompiledGraph (has .invoke and .stream)."""
        from src.agent import build_agent
        agent = build_agent()
        assert callable(getattr(agent, "invoke", None)), "Agent must have .invoke"
        assert callable(getattr(agent, "stream", None)), "Agent must have .stream"

    def test_system_prompt_schema_first(self):
        """System prompt must instruct the agent to call get_database_schema first."""
        from src.agent import _SYSTEM_PROMPT
        assert "get_database_schema" in _SYSTEM_PROMPT
        assert "ALWAYS" in _SYSTEM_PROMPT or "MANDATORY" in _SYSTEM_PROMPT

    def test_system_prompt_portuguese(self):
        """System prompt must instruct responses in Brazilian Portuguese."""
        from src.agent import _SYSTEM_PROMPT
        assert "pt-BR" in _SYSTEM_PROMPT or "Portuguese" in _SYSTEM_PROMPT or "português" in _SYSTEM_PROMPT.lower()

    def test_system_prompt_security(self):
        """System prompt must mention SELECT-only restriction."""
        from src.agent import _SYSTEM_PROMPT
        assert "SELECT" in _SYSTEM_PROMPT
        assert any(word in _SYSTEM_PROMPT for word in ("INSERT", "DELETE", "DROP"))

    def test_agent_has_two_tools(self):
        """
        Verify that both expected tools are registered in the module before
        the agent is built. We inspect the tool objects directly since
        LangGraph's compiled graph doesn't expose them via a public attribute.
        """
        from src.tools import execute_sql_query, get_database_schema
        assert get_database_schema.name == "get_database_schema"
        assert execute_sql_query.name == "execute_sql_query"
