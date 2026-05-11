from __future__ import annotations

import os
from typing import Iterator

from backend.llm.provider_interface import BaseLLMProvider
from backend.llm.validators import LLMRequest, LLMResponse, LLMStreamChunk, ProviderError


class OpenAIProvider(BaseLLMProvider):
    """OpenAI via the official `openai` SDK. Default model: gpt-4.1-mini."""

    name = "openai"
    default_model = "gpt-4.1-mini"

    def __init__(self, api_key: str | None = None, default_model: str | None = None) -> None:
        from openai import OpenAI

        self._OpenAI = OpenAI
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self._api_key:
            raise ProviderError(self.name, "OPENAI_API_KEY not set")
        self._client = OpenAI(api_key=self._api_key)
        self.default_model = default_model or self.default_model

    def _build_messages(self, request: LLMRequest) -> list[dict]:
        messages: list[dict] = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        for m in request.messages:
            messages.append({"role": m.role, "content": m.content})
        return messages

    def generate(self, request: LLMRequest) -> LLMResponse:
        kwargs: dict = {
            "model": request.model or self.default_model,
            "messages": self._build_messages(request),
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
        }
        if request.response_format == "json":
            kwargs["response_format"] = {"type": "json_object"}
        try:
            response = self._client.chat.completions.create(**kwargs)
        except Exception as exc:
            raise ProviderError(self.name, str(exc), exc) from exc

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

        client = AsyncOpenAI(api_key=self._api_key)
        kwargs: dict = {
            "model": request.model or self.default_model,
            "messages": self._build_messages(request),
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
        }
        if request.response_format == "json":
            kwargs["response_format"] = {"type": "json_object"}
        try:
            response = await client.chat.completions.create(**kwargs)
        except Exception as exc:
            raise ProviderError(self.name, str(exc), exc) from exc
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
        kwargs: dict = {
            "model": request.model or self.default_model,
            "messages": self._build_messages(request),
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "stream": True,
        }
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
            raise ProviderError(self.name, str(exc), exc) from exc
