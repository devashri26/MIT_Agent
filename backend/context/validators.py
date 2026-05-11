from __future__ import annotations

from pydantic import BaseModel, Field


class Citation(BaseModel):
    chunk_id: str
    source_url: str
    title: str
    section_path: list[str] = Field(default_factory=list)


class ContextBlock(BaseModel):
    chunk_id: str
    document_id: str
    text: str
    citation: Citation
    source_url: str
    title: str
    page_type: str
    section_type: str
    section_path: list[str] = Field(default_factory=list)
    rerank_score: float = 0.0
    answerability_score: float = 0.0
    final_relevance: float = 0.0
    token_count: int = 0


class DroppedBlock(BaseModel):
    chunk_id: str
    reason: str
    token_count: int = 0


class GroundedContext(BaseModel):
    query: str
    intent: str
    context_blocks: list[ContextBlock] = Field(default_factory=list)
    grounding_confidence: float = 0.0
    grounding_warnings: list[str] = Field(default_factory=list)
    total_tokens: int = 0
    token_budget: int = 0
    distinct_section_types: int = 0
    distinct_documents: int = 0
    prompt: str = ""
    dropped_blocks: list[DroppedBlock] = Field(default_factory=list)


class ContextBuildRequest(BaseModel):
    query: str
    top_k: int = 5
    candidate_pool: int = 20
    token_budget: int = 2000
    min_grounding_confidence: float = 0.5
    min_blocks: int = 2
    include_components: bool = False
