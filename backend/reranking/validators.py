from __future__ import annotations

from pydantic import Field

from backend.retrieval.models.search import RetrievedChunk, SearchResponse


class RerankedChunk(RetrievedChunk):
    rerank_score_raw: float = 0.0
    rerank_score: float = 0.0
    answerability_score: float = 0.0
    final_relevance: float = 0.0
    duplicate_of: str | None = None
    diversity_kept: bool = True
    rejection_reason: str | None = None


class RerankedSearchResponse(SearchResponse):
    rerank_model: str = ""
    candidate_pool: int = 0
    results: list[RerankedChunk] = Field(default_factory=list)
    rejected: list[RerankedChunk] = Field(default_factory=list)
