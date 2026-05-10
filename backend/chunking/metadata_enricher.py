import hashlib
from typing import Any

from backend.chunking.models.chunk import ChunkMetadata, SemanticChunk


class MetadataEnricher:
    def build_chunk(
        self,
        document: dict[str, Any],
        section_heading: str,
        section_path: list[str],
        chunk_index: int,
        text: str,
        token_count: int,
        content_type: str,
        warnings: list[str],
    ) -> SemanticChunk:
        metadata = document.get("metadata") or {}
        chunk_hash = self.chunk_hash(text)
        source_quality = self._source_quality(metadata.get("quality_warnings") or [], metadata.get("validation_issues") or [])
        return SemanticChunk(
            chunk_id=f"{document['page_id']}:{chunk_index:04d}:{chunk_hash[:12]}",
            document_id=str(document["page_id"]),
            url=str(document["url"]),
            title=str(document["title"]),
            department=document.get("department"),
            page_type=str(document.get("page_type") or "General"),
            section_heading=section_heading,
            chunk_index=chunk_index,
            text=text,
            token_count=token_count,
            content_type=content_type,
            quality_score=max(0.0, source_quality - (0.1 * len(warnings))),
            chunk_hash=chunk_hash,
            metadata=ChunkMetadata(
                section_path=section_path,
                headings=list(metadata.get("headings") or []),
                source_quality=source_quality,
                validation_issues=list(metadata.get("validation_issues") or []),
                quality_warnings=sorted(set([*list(metadata.get("quality_warnings") or []), *warnings])),
            ),
        )

    @staticmethod
    def chunk_hash(text: str) -> str:
        normalized = " ".join(text.lower().split())
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    @staticmethod
    def _source_quality(quality_warnings: list[str], validation_issues: list[dict[str, Any]]) -> float:
        score = 1.0
        score -= min(0.4, len(quality_warnings) * 0.08)
        score -= min(0.3, len(validation_issues) * 0.05)
        return round(max(0.0, score), 3)

