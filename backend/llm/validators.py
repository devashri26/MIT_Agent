from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


Role = Literal["system", "user", "assistant"]


class LLMMessage(BaseModel):
    role: Role
    content: str


class LLMRequest(BaseModel):
    system_prompt: str = ""
    messages: list[LLMMessage] = Field(default_factory=list)
    model: str = ""
    temperature: float = 0.0
    max_tokens: int = 1024
    response_format: Literal["text", "json"] = "text"
    timeout_seconds: float = 30.0


class LLMResponse(BaseModel):
    text: str
    model: str
    provider: str
    finish_reason: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    raw: dict[str, Any] = Field(default_factory=dict)


class LLMStreamChunk(BaseModel):
    delta: str = ""
    done: bool = False
    finish_reason: str = ""


class ProviderError(Exception):
    """Provider-neutral error raised when the underlying SDK fails. Carries the provider
    name and (optionally) the original exception for retry/logging."""

    def __init__(self, provider: str, message: str, original: Exception | None = None) -> None:
        super().__init__(f"[{provider}] {message}")
        self.provider = provider
        self.original = original
