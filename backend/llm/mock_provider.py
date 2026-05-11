from __future__ import annotations

from typing import Iterator

from backend.llm.provider_interface import BaseLLMProvider
from backend.llm.validators import LLMRequest, LLMResponse, LLMStreamChunk


class MockLLMProvider(BaseLLMProvider):
    """Deterministic provider for tests and key-free local dev. Returns the user-message
    plus a canned grounded suffix; honors stream() chunk-by-chunk so streaming code
    paths get exercised."""

    name = "mock"
    default_model = "mock-grounded"

    def __init__(self, canned_response: str | None = None) -> None:
        self.canned_response = canned_response

    def _compose_response(self, request: LLMRequest) -> str:
        if self.canned_response is not None:
            return self.canned_response
        last_user = next(
            (m.content for m in reversed(request.messages) if m.role == "user"), ""
        )
        return f"[mock answer] {last_user.strip()}"

    def generate(self, request: LLMRequest) -> LLMResponse:
        text = self._compose_response(request)
        return LLMResponse(
            text=text,
            model=request.model or self.default_model,
            provider=self.name,
            finish_reason="stop",
            output_tokens=len(text.split()),
        )

    async def generate_async(self, request: LLMRequest) -> LLMResponse:
        return self.generate(request)

    def stream(self, request: LLMRequest) -> Iterator[LLMStreamChunk]:
        text = self._compose_response(request)
        for word in text.split():
            yield LLMStreamChunk(delta=word + " ", done=False)
        yield LLMStreamChunk(delta="", done=True, finish_reason="stop")
