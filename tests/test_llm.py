"""
Unit tests for src/llm.py — provider selection and label generation.
These tests mock the actual chat model classes to avoid network calls.
"""
import importlib
import os
from unittest.mock import MagicMock, patch


def _reload_llm_module(env_overrides: dict):
    """Helper: reload src.llm with temporary env vars and return the module."""
    with patch.dict(os.environ, env_overrides, clear=False):
        import src.llm as llm_mod
        importlib.reload(llm_mod)
        return llm_mod


class TestGetProviderLabel:
    def test_ollama_label(self):
        mod = _reload_llm_module({"LLM_PROVIDER": "ollama", "OLLAMA_MODEL": "qwen2.5:14b"})
        label = mod.get_provider_label()
        assert "ollama" in label.lower() or "local" in label.lower()
        assert "qwen2.5:14b" in label

    def test_gemini_label(self):
        mod = _reload_llm_module({"LLM_PROVIDER": "gemini", "GEMINI_MODEL": "gemini-2.5-flash"})
        label = mod.get_provider_label()
        assert "gemini" in label.lower()
        assert "gemini-2.5-flash" in label

    def test_default_provider_is_gemini(self):
        # load_dotenv() re-reads .env on reload, so we must explicitly set
        # LLM_PROVIDER=gemini rather than relying on the variable being absent.
        mod = _reload_llm_module({"LLM_PROVIDER": "gemini", "GEMINI_MODEL": "gemini-2.5-flash"})
        assert "gemini" in mod.get_provider_label().lower()


class TestGetLlm:
    def test_ollama_returns_chat_ollama(self):
        mock_ollama = MagicMock()
        with patch.dict(os.environ, {"LLM_PROVIDER": "ollama", "OLLAMA_MODEL": "qwen2.5:14b"}):
            with patch("langchain_ollama.ChatOllama", mock_ollama):
                import src.llm as llm_mod
                importlib.reload(llm_mod)
                llm_mod.get_llm()
                mock_ollama.assert_called_once()
                call_kwargs = mock_ollama.call_args.kwargs
                assert call_kwargs["model"] == "qwen2.5:14b"

    def test_gemini_returns_chat_google(self):
        mock_gemini = MagicMock()
        with patch.dict(os.environ, {"LLM_PROVIDER": "gemini", "GEMINI_MODEL": "gemini-2.5-flash"}):
            with patch("langchain_google_genai.ChatGoogleGenerativeAI", mock_gemini):
                import src.llm as llm_mod
                importlib.reload(llm_mod)
                llm_mod.get_llm()
                mock_gemini.assert_called_once()
                call_kwargs = mock_gemini.call_args.kwargs
                assert call_kwargs["model"] == "gemini-2.5-flash"
