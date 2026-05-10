from typing import Any

from pydantic import BaseModel, Field


class ChunkMetadata(BaseModel):
    section_path: list[str] = Field(default_factory=list)
    headings: list[str] = Field(default_factory=list)
    source_quality: float = 0.0
    validation_issues: list[dict[str, Any]] = Field(default_factory=list)
    quality_warnings: list[str] = Field(default_factory=list)


class SemanticChunk(BaseModel):
    chunk_id: str
    document_id: str
    url: str
    title: str
    department: str | None
    page_type: str
    section_heading: str
    chunk_index: int
    text: str
    token_count: int
    content_type: str
    quality_score: float
    chunk_hash: str
    metadata: ChunkMetadata


class ChunkingReport(BaseModel):
    documents_processed: int = 0
    chunks_generated: int = 0
    avg_chunk_tokens: float = 0.0
    max_chunk_tokens: int = 0
    min_chunk_tokens: int = 0
    content_type_distribution: dict[str, int] = Field(default_factory=dict)
    rejected_chunks: int = 0
    boilerplate_removed: int = 0
    tiny_chunks_merged: int = 0
