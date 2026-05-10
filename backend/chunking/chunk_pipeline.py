from collections import Counter
from pathlib import Path
from typing import Any

import orjson

from backend.chunking.content_type_detector import ContentTypeDetector
from backend.chunking.metadata_enricher import MetadataEnricher
from backend.chunking.models.chunk import ChunkingReport, SemanticChunk
from backend.chunking.section_splitter import SectionSplitter
from backend.chunking.semantic_grouper import SemanticGrouper
from backend.chunking.serializers import ChunkSerializer
from backend.chunking.token_splitter import TokenSplitter
from backend.chunking.validators import ChunkValidator


class ChunkPipeline:
    def __init__(
        self,
        section_splitter: SectionSplitter | None = None,
        token_splitter: TokenSplitter | None = None,
        content_type_detector: ContentTypeDetector | None = None,
        validator: ChunkValidator | None = None,
        enricher: MetadataEnricher | None = None,
        serializer: ChunkSerializer | None = None,
    ) -> None:
        self.section_splitter = section_splitter or SectionSplitter()
        self.token_splitter = token_splitter or TokenSplitter()
        self.semantic_grouper = SemanticGrouper(self.token_splitter)
        self.content_type_detector = content_type_detector or ContentTypeDetector()
        self.validator = validator or ChunkValidator()
        self.enricher = enricher or MetadataEnricher()
        self.serializer = serializer or ChunkSerializer()

    def run(
        self,
        input_path: Path = Path("datasets/processed_documents.json"),
        output_path: Path = Path("datasets/chunks.jsonl"),
        report_path: Path = Path("reports/chunking_report.json"),
    ) -> ChunkingReport:
        documents = self._load_documents(input_path)
        chunks, report = self.chunk_documents(documents)
        self.serializer.write_jsonl(chunks, output_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_bytes(orjson.dumps(report.model_dump(mode="json"), option=orjson.OPT_INDENT_2))
        return report

    def chunk_documents(self, documents: list[dict[str, Any]]) -> tuple[list[SemanticChunk], ChunkingReport]:
        chunks: list[SemanticChunk] = []
        content_type_counts: Counter[str] = Counter()
        rejected_chunks = 0
        boilerplate_removed = 0
        tiny_chunks_merged = 0
        duplicate_hashes: set[str] = set()

        for document in documents:
            chunk_index = 0
            sections = self.section_splitter.split(document)
            for section in sections:
                heading = section["heading"]
                content = section["content"]
                content_type = self.content_type_detector.detect(str(document.get("page_type") or "General"), heading, content)
                groups = self.semantic_grouper.group(heading, content, content_type)
                candidate_texts: list[str] = []
                for group in groups:
                    candidate_texts.extend(self.token_splitter.split_text(group))
                candidate_texts, merged_count = self._merge_tiny_chunks(candidate_texts)
                tiny_chunks_merged += merged_count
                for text in candidate_texts:
                    if self.validator.should_reject(text):
                        rejected_chunks += 1
                        boilerplate_removed += 1
                        continue
                    token_count = self.token_splitter.count_tokens(text)
                    if token_count < 40:
                        text = self._contextualize_tiny_chunk(document, heading, content_type, text)
                        token_count = self.token_splitter.count_tokens(text)
                    warning_list = self.validator.validate_text(text, token_count)
                    chunk = self.enricher.build_chunk(
                        document=document,
                        section_heading=heading,
                        section_path=[heading],
                        chunk_index=chunk_index,
                        text=text,
                        token_count=token_count,
                        content_type=content_type,
                        warnings=warning_list,
                    )
                    chunk_warnings = self.validator.validate(chunk)
                    if chunk.chunk_hash in duplicate_hashes:
                        chunk_warnings.append("duplicate_chunk_hash")
                    if chunk_warnings:
                        chunk.metadata.quality_warnings = sorted(set([*chunk.metadata.quality_warnings, *chunk_warnings]))
                        chunk.quality_score = max(0.0, round(chunk.quality_score - (0.1 * len(chunk_warnings)), 3))
                    duplicate_hashes.add(chunk.chunk_hash)
                    chunks.append(chunk)
                    content_type_counts[content_type] += 1
                    chunk_index += 1

        token_counts = [chunk.token_count for chunk in chunks]
        report = ChunkingReport(
            documents_processed=len(documents),
            chunks_generated=len(chunks),
            avg_chunk_tokens=round(sum(token_counts) / len(token_counts), 2) if token_counts else 0.0,
            max_chunk_tokens=max(token_counts) if token_counts else 0,
            min_chunk_tokens=min(token_counts) if token_counts else 0,
            content_type_distribution=dict(sorted(content_type_counts.items())),
            rejected_chunks=rejected_chunks,
            boilerplate_removed=boilerplate_removed,
            tiny_chunks_merged=tiny_chunks_merged,
        )
        return chunks, report

    @staticmethod
    def _load_documents(input_path: Path) -> list[dict[str, Any]]:
        return orjson.loads(input_path.read_bytes())

    def _merge_tiny_chunks(self, texts: list[str], min_tokens: int = 40) -> tuple[list[str], int]:
        if len(texts) <= 1:
            return texts, 0
        merged: list[str] = []
        tiny_buffer: list[str] = []
        merged_count = 0

        for text in texts:
            token_count = self.token_splitter.count_tokens(text)
            if token_count < min_tokens:
                tiny_buffer.append(text)
                if self.token_splitter.count_tokens("\n\n".join(tiny_buffer)) >= min_tokens:
                    merged.append("\n\n".join(tiny_buffer))
                    merged_count += max(0, len(tiny_buffer) - 1)
                    tiny_buffer = []
                continue
            if tiny_buffer:
                combined = "\n\n".join([*tiny_buffer, text])
                if self.token_splitter.count_tokens(combined) <= self.token_splitter.hard_max:
                    merged.append(combined)
                    merged_count += len(tiny_buffer)
                else:
                    merged.extend(tiny_buffer)
                    merged.append(text)
                tiny_buffer = []
            else:
                merged.append(text)

        if tiny_buffer:
            if merged:
                combined = "\n\n".join([merged[-1], *tiny_buffer])
                if self.token_splitter.count_tokens(combined) <= self.token_splitter.hard_max:
                    merged[-1] = combined
                    merged_count += len(tiny_buffer)
                else:
                    merged.extend(tiny_buffer)
            else:
                merged.append("\n\n".join(tiny_buffer))
                merged_count += max(0, len(tiny_buffer) - 1)
        return merged, merged_count

    @staticmethod
    def _contextualize_tiny_chunk(document: dict[str, Any], heading: str, content_type: str, text: str) -> str:
        context = [
            f"Title: {document.get('title') or 'Untitled'}",
            f"URL: {document.get('url') or ''}",
            f"Page type: {document.get('page_type') or 'General'}",
            f"Content type: {content_type}",
            f"Section: {heading}",
        ]
        department = document.get("department")
        if department:
            context.append(f"Department: {department}")
        return "\n".join([*context, "", text.strip()])
