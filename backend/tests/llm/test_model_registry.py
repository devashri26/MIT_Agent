import os

import pytest

from backend.llm.model_registry import (
    DEFAULT_MODELS,
    resolve_default_model,
    resolve_default_provider,
)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for var in ("LLM_PROVIDER", "GOOGLE_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(var, raising=False)


def test_resolve_default_provider_fallbacks_to_mock_when_no_key(monkeypatch) -> None:
    assert resolve_default_provider() == "mock"


def test_resolve_default_provider_respects_explicit_env(monkeypatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    assert resolve_default_provider() == "gemini"


def test_resolve_default_provider_uses_key_when_present(monkeypatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    assert resolve_default_provider() == "anthropic"


def test_resolve_default_model_has_entries_for_each_provider() -> None:
    for provider in ["gemini", "anthropic", "openai", "mock"]:
        assert resolve_default_model(provider) == DEFAULT_MODELS[provider]
