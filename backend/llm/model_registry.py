from __future__ import annotations

import os


DEFAULT_MODELS: dict[str, str] = {
    # 8b-instant has 14,400 RPD on Groq free tier (vs 1,000 for 70b) plus higher TPM
    # headroom. Quality is slightly lower but grounded RAG doesn't need 70b reasoning.
    # Pass "model": "llama-3.3-70b-versatile" in the request body if you specifically
    # want the bigger model and you're OK with tighter limits.
    "groq": "llama-3.1-8b-instant",
    "gemini": "gemini-2.5-flash",
    "anthropic": "claude-sonnet-4-6",
    "openai": "gpt-4.1-mini",
    "mock": "mock-grounded",
}


# Order matters — first provider with a key set in env wins when LLM_PROVIDER isn't explicit.
PROVIDER_ENV_VARS: dict[str, str] = {
    "groq": "GROQ_API_KEY",
    "gemini": "GOOGLE_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
}


DEFAULT_PROVIDER_ENV_VAR = "LLM_PROVIDER"
DEFAULT_PROVIDER = "groq"


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
