from __future__ import annotations

import os
from typing import Iterator

from backend.llm.provider_interface import BaseLLMProvider
from backend.llm.validators import LLMRequest, LLMResponse, LLMStreamChunk, ProviderError


class GeminiProvider(BaseLLMProvider):
    """Google Gemini via the `google-genai` SDK. Default model: gemini-2.5-flash."""

    name = "gemini"
    default_model = "gemini-2.5-flash"

    def __init__(self, api_key: str | None = None, default_model: str | None = None) -> None:
        from google import genai

        self._genai = genai
        self._api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not self._api_key:
            raise ProviderError(self.name, "GOOGLE_API_KEY not set")
        self._client = genai.Client(api_key=self._api_key)
        self.default_model = default_model or self.default_model

    def _build_contents(self, request: LLMRequest) -> list[dict]:
        contents: list[dict] = []
        for message in request.messages:
            role = "user" if message.role in ("user", "system") else "model"
            contents.append({"role": role, "parts": [{"text": message.content}]})
        return contents

    def _config(self, request: LLMRequest) -> dict:
        cfg: dict = {
            "temperature": request.temperature,
            "max_output_tokens": request.max_tokens,
        }
        if request.system_prompt:
            cfg["system_instruction"] = request.system_prompt
        if request.response_format == "json":
            cfg["response_mime_type"] = "application/json"
        return cfg

    def generate(self, request: LLMRequest) -> LLMResponse:
        try:
            response = self._client.models.generate_content(
                model=request.model or self.default_model,
                contents=self._build_contents(request),
                config=self._config(request),
            )
        except Exception as exc:
            raise ProviderError(self.name, str(exc), exc) from exc

        text = response.text or ""
        usage = getattr(response, "usage_metadata", None)
        return LLMResponse(
            text=text,
            model=request.model or self.default_model,
            provider=self.name,
            finish_reason=getattr(response, "finish_reason", "") or "",
            input_tokens=int(getattr(usage, "prompt_token_count", 0) or 0) if usage else 0,
            output_tokens=int(getattr(usage, "candidates_token_count", 0) or 0) if usage else 0,
        )

    async def generate_async(self, request: LLMRequest) -> LLMResponse:
        try:
            response = await self._client.aio.models.generate_content(
                model=request.model or self.default_model,
                contents=self._build_contents(request),
                config=self._config(request),
            )
        except Exception as exc:
            raise ProviderError(self.name, str(exc), exc) from exc
        text = response.text or ""
        usage = getattr(response, "usage_metadata", None)
        return LLMResponse(
            text=text,
            model=request.model or self.default_model,
            provider=self.name,
            finish_reason=getattr(response, "finish_reason", "") or "",
            input_tokens=int(getattr(usage, "prompt_token_count", 0) or 0) if usage else 0,
            output_tokens=int(getattr(usage, "candidates_token_count", 0) or 0) if usage else 0,
        )

    def stream(self, request: LLMRequest) -> Iterator[LLMStreamChunk]:
        try:
            stream = self._client.models.generate_content_stream(
                model=request.model or self.default_model,
                contents=self._build_contents(request),
                config=self._config(request),
            )
            for event in stream:
                delta = event.text or ""
                if delta:
                    yield LLMStreamChunk(delta=delta, done=False)
            yield LLMStreamChunk(delta="", done=True, finish_reason="stop")
        except Exception as exc:
            raise ProviderError(self.name, str(exc), exc) from exc
