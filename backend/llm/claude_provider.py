from __future__ import annotations

import os
from typing import Iterator

from backend.llm.provider_interface import BaseLLMProvider
from backend.llm.validators import LLMRequest, LLMResponse, LLMStreamChunk, ProviderError


class ClaudeProvider(BaseLLMProvider):
    """Anthropic Claude via the official `anthropic` SDK. Default model: claude-sonnet-4-6."""

    name = "anthropic"
    default_model = "claude-sonnet-4-6"

    def __init__(self, api_key: str | None = None, default_model: str | None = None) -> None:
        from anthropic import Anthropic

        self._Anthropic = Anthropic
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self._api_key:
            raise ProviderError(self.name, "ANTHROPIC_API_KEY not set")
        self._client = Anthropic(api_key=self._api_key)
        self.default_model = default_model or self.default_model

    def _build_messages(self, request: LLMRequest) -> list[dict]:
        return [{"role": m.role, "content": m.content} for m in request.messages if m.role != "system"]

    def _kwargs(self, request: LLMRequest) -> dict:
        kwargs: dict = {
            "model": request.model or self.default_model,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "messages": self._build_messages(request),
        }
        if request.system_prompt:
            kwargs["system"] = request.system_prompt
        return kwargs

    def generate(self, request: LLMRequest) -> LLMResponse:
        try:
            response = self._client.messages.create(**self._kwargs(request))
        except Exception as exc:
            raise ProviderError(self.name, str(exc), exc) from exc
        text_blocks = [block.text for block in response.content if getattr(block, "type", "") == "text"]
        return LLMResponse(
            text="".join(text_blocks),
            model=response.model,
            provider=self.name,
            finish_reason=response.stop_reason or "",
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )

    async def generate_async(self, request: LLMRequest) -> LLMResponse:
        from anthropic import AsyncAnthropic

        client = AsyncAnthropic(api_key=self._api_key)
        try:
            response = await client.messages.create(**self._kwargs(request))
        except Exception as exc:
            raise ProviderError(self.name, str(exc), exc) from exc
        text_blocks = [block.text for block in response.content if getattr(block, "type", "") == "text"]
        return LLMResponse(
            text="".join(text_blocks),
            model=response.model,
            provider=self.name,
            finish_reason=response.stop_reason or "",
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )

    def stream(self, request: LLMRequest) -> Iterator[LLMStreamChunk]:
        try:
            with self._client.messages.stream(**self._kwargs(request)) as stream:
                for text in stream.text_stream:
                    if text:
                        yield LLMStreamChunk(delta=text, done=False)
                yield LLMStreamChunk(delta="", done=True, finish_reason="stop")
        except Exception as exc:
            raise ProviderError(self.name, str(exc), exc) from exc
