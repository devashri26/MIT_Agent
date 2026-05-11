from typing import Any

from backend.normalization.canonicalizer import CanonicalIndex
from backend.normalization.page_classifier import DeterministicPageClassifier
from backend.normalization.quality_flags import compute_quality_flags
from backend.normalization.retrieval_priority import compute_retrieval_priority
from backend.normalization.semantic_section_typer import SemanticSectionTyper
from backend.normalization.validators import NormalizedChunk


class MetadataNormalizer:
    """Per-chunk orchestrator: classify, type, flag, score, register canonical state."""

    def __init__(self, canonical_index: CanonicalIndex | None = None) -> None:
        self.classifier = DeterministicPageClassifier()
        self.section_typer = SemanticSectionTyper()
        self.canonical_index = canonical_index or CanonicalIndex()

    def normalize_chunk(
        self,
        chunk: dict[str, Any],
        document_headings: list[str] | None = None,
        document_content: str = "",
    ) -> NormalizedChunk:
        url = chunk.get("url", "")
        canonical_url, is_canonical = self.canonical_index.register_url(url)
        is_unique_hash = self.canonical_index.register_chunk_hash(chunk.get("chunk_hash", ""))

        chunk_metadata = chunk.get("metadata") or {}
        classifier_headings = document_headings or chunk_metadata.get("headings") or []

        page_type, page_type_confidence = self.classifier.classify(
            url=url,
            title=chunk.get("title", ""),
            headings=classifier_headings,
            content=document_content or chunk.get("text", ""),
        )

        section_type = self.section_typer.type_section(
            section_heading=chunk.get("section_heading", ""),
            headings=chunk_metadata.get("headings") or [],
            text=chunk.get("text", ""),
            page_type=page_type,
        )

        quality_flags = compute_quality_flags(
            text=chunk.get("text", ""),
            token_count=chunk.get("token_count", 0),
            page_type_confidence=page_type_confidence,
            page_type=page_type,
            quality_score=chunk.get("quality_score", 0.0),
        )
        if not is_unique_hash:
            quality_flags.append("duplicate")
        if not is_canonical:
            quality_flags.append("non_canonical")

        retrieval_priority = compute_retrieval_priority(
            page_type=page_type,
            quality_flags=quality_flags,
            page_type_confidence=page_type_confidence,
            quality_score=chunk.get("quality_score", 0.0),
        )

        return NormalizedChunk(
            chunk_id=chunk["chunk_id"],
            document_id=chunk["document_id"],
            url=url,
            canonical_url=canonical_url,
            title=chunk.get("title", ""),
            department=chunk.get("department"),
            section_heading=chunk.get("section_heading", ""),
            chunk_index=chunk.get("chunk_index", 0),
            text=chunk.get("text", ""),
            token_count=chunk.get("token_count", 0),
            content_type=chunk.get("content_type", ""),
            quality_score=chunk.get("quality_score", 0.0),
            chunk_hash=chunk.get("chunk_hash", ""),
            page_type=page_type,
            page_type_confidence=round(page_type_confidence, 3),
            section_type=section_type,
            retrieval_priority=retrieval_priority,
            quality_flags=quality_flags,
            is_canonical=is_canonical,
            metadata=chunk_metadata,
        )
