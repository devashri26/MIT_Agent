from __future__ import annotations

import os


DEFAULT_MODELS: dict[str, str] = {
    "gemini": "gemini-2.5-flash",
    "anthropic": "claude-sonnet-4-6",
    "openai": "gpt-4.1-mini",
    "mock": "mock-grounded",
}


PROVIDER_ENV_VARS: dict[str, str] = {
    "gemini": "GOOGLE_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
}


DEFAULT_PROVIDER_ENV_VAR = "LLM_PROVIDER"
DEFAULT_PROVIDER = "gemini"


def resolve_default_provider() -> str:
    """Determine which provider to instantiate by default.

    Precedence: explicit LLM_PROVIDER env var > provider with API key set in env >
    fallback to 'mock' (so the system stays runnable without any key)."""
    explicit = os.environ.get(DEFAULT_PROVIDER_ENV_VAR, "").strip().lower()
    if explicit in DEFAULT_MODELS:
        return explicit
    for name, env_var in PROVIDER_ENV_VARS.items():
        if os.environ.get(env_var):
            return name
    return "mock"


def resolve_default_model(provider: str) -> str:
    return DEFAULT_MODELS.get(provider, DEFAULT_MODELS["mock"])
