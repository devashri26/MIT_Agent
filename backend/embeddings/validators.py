from __future__ import annotations

from pydantic import BaseModel, Field


class EmbeddedChunk(BaseModel):
    chunk_id: str
    chunk_hash: str
    embedding: list[float]
    embedding_dimension: int
    embedded_at: str
    embedding_model: str


class EmbeddingManifest(BaseModel):
    embedding_model: str
    embedding_dimension: int
    total_eligible_chunks: int = 0
    total_embedded: int = 0
    skipped_reusable_components: int = 0
    skipped_contaminated: int = 0
    skipped_cta_heavy: int = 0
    skipped_boilerplate_heavy: int = 0
    cache_hits: int = 0
    embedded_at: str = ""
    output_path: str = ""
    skip_reason_distribution: dict[str, int] = Field(default_factory=dict)
