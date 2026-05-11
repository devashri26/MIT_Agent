import pytest

from backend.llm.groq_provider import GroqProvider
from backend.llm.validators import ProviderError


def test_groq_provider_requires_api_key(monkeypatch) -> None:
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(ProviderError) as info:
        GroqProvider()
    assert "GROQ_API_KEY" in str(info.value)


def test_groq_provider_default_model() -> None:
    assert GroqProvider.default_model == "llama-3.3-70b-versatile"
    assert GroqProvider.name == "groq"
