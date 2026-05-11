from typing import Any

from pydantic import BaseModel, Field

from backend.retrieval.retrieval_debugger import RetrievalExplanation


class RetrievedChunk(BaseModel):
    rank: int
    score: float
    chunk_id: str
    document_id: str
    url: str
    canonical_url: str
    title: str
    department: str | None = None
    page_type: str
    section_type: str
    section_heading: str
    section_path: list[str] = Field(default_factory=list)
    token_count: int
    content_type: str
    quality_score: float
    retrieval_priority: float
    quality_flags: list[str] = Field(default_factory=list)
    is_reusable_component: bool = False
    component_type: str | None = None
    mixed_topic: bool = False
    dominant_topics: list[str] = Field(default_factory=list)
    cross_domain_contamination: bool = False
    contamination_sources: list[str] = Field(default_factory=list)
    retrieval_source: list[str] = Field(default_factory=list)
    bm25_rank: int = 0
    dense_rank: int = 0
    fusion_score: float = 0.0
    text: str
    explanation: RetrievalExplanation
    metadata: dict[str, Any] = Field(default_factory=dict)


class SearchResponse(BaseModel):
    query: str
    top_k: int
    intent: str
    allowed_page_types: list[str] = Field(default_factory=list)
    allowed_section_types: list[str] = Field(default_factory=list)
    expanded_terms: list[str] = Field(default_factory=list)
    filter_fallback_used: bool = False
    components_excluded: int = 0
    results: list[RetrievedChunk] = Field(default_factory=list)
