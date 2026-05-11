from __future__ import annotations

import orjson

from backend.llm.validators import LLMStreamChunk


def to_sse_line(chunk: LLMStreamChunk) -> bytes:
    """Encode an LLMStreamChunk as one SSE message line."""
    payload = chunk.model_dump(mode="json")
    return b"data: " + orjson.dumps(payload) + b"\n\n"
