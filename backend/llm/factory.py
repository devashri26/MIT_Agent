from __future__ import annotations

from backend.llm.model_registry import resolve_default_provider
from backend.llm.provider_interface import BaseLLMProvider


_provider_cache: dict[str, BaseLLMProvider] = {}


def _construct(name: str) -> BaseLLMProvider:
    if name == "groq":
        from backend.llm.groq_provider import GroqProvider

        return GroqProvider()
    if name == "gemini":
        from backend.llm.gemini_provider import GeminiProvider

        return GeminiProvider()
    if name == "anthropic":
        from backend.llm.claude_provider import ClaudeProvider

        return ClaudeProvider()
    if name == "openai":
        from backend.llm.openai_provider import OpenAIProvider

        return OpenAIProvider()
    from backend.llm.mock_provider import MockLLMProvider

    return MockLLMProvider()


def get_provider(name: str | None = None) -> BaseLLMProvider:
    """Cached provider construction. The SDK client (and any HTTP keep-alive pool it owns)
    is reused across requests. Falls back to MockLLMProvider when initialization fails."""
    resolved = (name or resolve_default_provider()).lower()
    if resolved in _provider_cache:
        return _provider_cache[resolved]
    try:
        provider = _construct(resolved)
    except Exception:
        provider = _construct("mock")
    _provider_cache[resolved] = provider
    return provider


def reset_provider_cache() -> None:
    """Test/dev helper — drop cached providers (use after rotating an API key)."""
    _provider_cache.clear()
