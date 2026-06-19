"""
Central LLM factory — reads LLM_PROVIDER from the environment and returns
the appropriate LangChain chat model.

Supported providers:
  gemini  — Google Gemini via langchain-google-genai (requires GOOGLE_API_KEY)
  ollama  — Local model via Ollama (no API key needed, requires Ollama running)
"""

import os

from dotenv import load_dotenv

load_dotenv()

_PROVIDER = os.getenv("LLM_PROVIDER", "gemini").lower()
_GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
_OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:14b")
_OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


def get_llm(temperature: float = 0):
    """
    Return a configured LangChain chat model based on LLM_PROVIDER.
    Lazy-imports the provider package so neither langchain-google-genai nor
    langchain-ollama is required unless actually used.
    """
    if _PROVIDER == "ollama":
        from langchain_ollama import ChatOllama  # noqa: PLC0415

        return ChatOllama(
            model=_OLLAMA_MODEL,
            base_url=_OLLAMA_BASE_URL,
            temperature=temperature,
        )

    # Default: Gemini
    from langchain_google_genai import ChatGoogleGenerativeAI  # noqa: PLC0415

    return ChatGoogleGenerativeAI(model=_GEMINI_MODEL, temperature=temperature)


def is_local_provider() -> bool:
    """Return True when using a local provider (Ollama) — tokens have no cost."""
    return _PROVIDER == "ollama"


def get_provider_label() -> str:
    """Human-readable label for the active provider (used in the UI)."""
    if _PROVIDER == "ollama":
        return f"🖥️ Local — {_OLLAMA_MODEL}"
    return f"☁️ Gemini — {_GEMINI_MODEL}"
