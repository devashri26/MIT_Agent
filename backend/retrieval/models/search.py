from typing import Any

from pydantic import BaseModel, Field


class RetrievedChunk(BaseModel):
    rank: int
    score: float
    chunk_id: str
    document_id: str
    url: str
    title: str
    department: str | None
    page_type: str
    section_heading: str
    token_count: int
    content_type: str
    quality_score: float
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class SearchResponse(BaseModel):
    query: str
    top_k: int
    results: list[RetrievedChunk]

