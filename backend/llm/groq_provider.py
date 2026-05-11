from __future__ import annotations

import os
from typing import Iterator

from backend.llm.provider_interface import BaseLLMProvider
from backend.llm.validators import LLMRequest, LLMResponse, LLMStreamChunk, ProviderError


GROQ_BASE_URL = "https://api.groq.com/openai/v1"


class GroqProvider(BaseLLMProvider):
    """Groq via the OpenAI-compatible API (uses the `openai` SDK with a Groq base_url).

    Default model is llama-3.1-8b-instant: 30 RPM, ~30K TPM, 14,400 RPD on free tier.
    Pass model='llama-3.3-70b-versatile' for higher quality (tighter limits: 12K TPM,
    1,000 RPD on free tier).
    """

    name = "groq"
    default_model = "llama-3.1-8b-instant"

    def __init__(self, api_key: str | None = None, default_model: str | None = None) -> None:
        from openai import OpenAI

        self._api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not self._api_key:
            raise ProviderError(self.name, "GROQ_API_KEY not set")
        self._client = OpenAI(api_key=self._api_key, base_url=GROQ_BASE_URL)
        self.default_model = default_model or self.default_model

    def _build_messages(self, request: LLMRequest) -> list[dict]:
        messages: list[dict] = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        for m in request.messages:
            messages.append({"role": m.role, "content": m.content})
        return messages

    def _kwargs(self, request: LLMRequest) -> dict:
        kwargs: dict = {
            "model": request.model or self.default_model,
            "messages": self._build_messages(request),
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
        }
        if request.response_format == "json":
            kwargs["response_format"] = {"type": "json_object"}
        return kwargs

    def generate(self, request: LLMRequest) -> LLMResponse:
        try:
            response = self._client.chat.completions.create(**self._kwargs(request))
        except Exception as exc:
            raise ProviderError(self.name, _friendly_error(exc), exc) from exc

        choice = response.choices[0]
        usage = response.usage
        return LLMResponse(
            text=choice.message.content or "",
            model=response.model,
            provider=self.name,
            finish_reason=choice.finish_reason or "",
            input_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
            output_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
        )

    async def generate_async(self, request: LLMRequest) -> LLMResponse:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=self._api_key, base_url=GROQ_BASE_URL)
        try:
            response = await client.chat.completions.create(**self._kwargs(request))
        except Exception as exc:
            raise ProviderError(self.name, _friendly_error(exc), exc) from exc
        choice = response.choices[0]
        usage = response.usage
        return LLMResponse(
            text=choice.message.content or "",
            model=response.model,
            provider=self.name,
            finish_reason=choice.finish_reason or "",
            input_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
            output_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
        )

    def stream(self, request: LLMRequest) -> Iterator[LLMStreamChunk]:
        kwargs = {**self._kwargs(request), "stream": True}
        try:
            stream = self._client.chat.completions.create(**kwargs)
            for event in stream:
                if not event.choices:
                    continue
                choice = event.choices[0]
                delta = (choice.delta.content or "") if choice.delta else ""
                if delta:
                    yield LLMStreamChunk(delta=delta, done=False)
            yield LLMStreamChunk(delta="", done=True, finish_reason="stop")
        except Exception as exc:
            raise ProviderError(self.name, _friendly_error(exc), exc) from exc


def _friendly_error(exc: Exception) -> str:
    text = str(exc)
    lower = text.lower()
    if "429" in text or "rate" in lower and "limit" in lower:
        return (
            "Groq rate limit hit (HTTP 429). Wait ~60s, switch to llama-3.1-8b-instant "
            "(higher RPM), or set LLM_PROVIDER=mock to test without quota."
        )
    if "401" in text or "invalid api key" in lower:
        return "Groq auth failed (HTTP 401). Check GROQ_API_KEY in your .env."
    if "model_not_found" in lower or "404" in text:
        return f"Groq model not found: {text}"
    return text
