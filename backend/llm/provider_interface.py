from __future__ import annotations

from abc import ABC, abstractmethod
from typing import AsyncIterator, Iterator

from backend.llm.validators import LLMRequest, LLMResponse, LLMStreamChunk


class BaseLLMProvider(ABC):
    """Provider-neutral LLM interface. Implementations must not leak provider-specific
    prompt formatting or response shapes — all callers deal in LLMRequest/LLMResponse."""

    name: str = ""
    default_model: str = ""

    @abstractmethod
    def generate(self, request: LLMRequest) -> LLMResponse:
        """Synchronous completion."""

    @abstractmethod
    async def generate_async(self, request: LLMRequest) -> LLMResponse:
        """Async completion. May internally call generate() in a threadpool if the SDK has
        no native async."""

    @abstractmethod
    def stream(self, request: LLMRequest) -> Iterator[LLMStreamChunk]:
        """Synchronous token stream. Final chunk has done=True."""

    async def stream_async(self, request: LLMRequest) -> AsyncIterator[LLMStreamChunk]:
        """Default: yield from the sync stream() in an async iterator. Subclasses with
        native async streaming should override."""
        for chunk in self.stream(request):
            yield chunk
