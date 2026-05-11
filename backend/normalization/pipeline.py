from collections import Counter
from pathlib import Path

import orjson

from backend.embeddings.eligibility import is_embedding_eligible
from backend.normalization.canonicalizer import CanonicalIndex
from backend.normalization.component_detector import build_registry
from backend.normalization.contamination_detector import detect_cross_domain_contamination
from backend.normalization.metadata_normalizer import MetadataNormalizer
from backend.normalization.retrieval_priority import compute_retrieval_priority
from backend.normalization.section_normalizer import normalize_section_path
from backend.normalization.semantic_section_splitter import detect_mixed_topic
from backend.normalization.validators import (
    CorpusNormalizationReport,
    SemanticNormalizationReport,
)
from backend.normalization.widget_suppressor import (
    assess_chunk_components,
    classify_registry,
)


def _build_document_lookup(documents_path: Path) -> dict[str, tuple[list[str], str]]:
    if not documents_path.exists():
        return {}
    documents = orjson.loads(documents_path.read_bytes())
    lookup: dict[str, tuple[list[str], str]] = {}
    for doc in documents:
        doc_id = doc.get("page_id")
        if not doc_id:
            continue
        headings = [section.get("heading", "") for section in doc.get("sections", [])]
        lookup[doc_id] = (headings, doc.get("clean_content", ""))
    return lookup


def _build_document_title_lookup(documents_path: Path) -> dict[str, str]:
    if not documents_path.exists():
        return {}
    documents = orjson.loads(documents_path.read_bytes())
    return {doc.get("page_id", ""): doc.get("title", "") for doc in documents}


def run_normalization(
    chunks_path: Path = Path("datasets/chunks.jsonl"),
    documents_path: Path = Path("datasets/processed_documents.json"),
    output_path: Path = Path("datasets/normalized_chunks.jsonl"),
    report_path: Path = Path("reports/corpus_normalization_report.json"),
) -> CorpusNormalizationReport:
    doc_lookup = _build_document_lookup(documents_path)
    normalizer = MetadataNormalizer(canonical_index=CanonicalIndex())

    page_type_counts: Counter[str] = Counter()
    section_type_counts: Counter[str] = Counter()
    quality_flag_counts: Counter[str] = Counter()
    canonical_pages = 0
    non_canonical_pages = 0
    weak_classifications = 0
    total_chunks = 0
    seen_documents: set[str] = set()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not chunks_path.exists():
        raise FileNotFoundError(f"chunks file not found: {chunks_path}")

    with output_path.open("wb") as out_file:
        for raw_line in chunks_path.read_bytes().splitlines():
            if not raw_line.strip():
                continue
            chunk = orjson.loads(raw_line)
            doc_id = chunk.get("document_id", "")
            headings, content = doc_lookup.get(doc_id, ([], ""))
            normalized = normalizer.normalize_chunk(chunk, headings, content)
            out_file.write(orjson.dumps(normalized.model_dump(mode="json")))
            out_file.write(b"\n")

            total_chunks += 1
            page_type_counts[normalized.page_type] += 1
            section_type_counts[normalized.section_type] += 1
            for flag in normalized.quality_flags:
                quality_flag_counts[flag] += 1
            if "weak_classification" in normalized.quality_flags:
                weak_classifications += 1
            if doc_id and doc_id not in seen_documents:
                seen_documents.add(doc_id)
                if normalized.is_canonical:
                    canonical_pages += 1
                else:
                    non_canonical_pages += 1

    report = CorpusNormalizationReport(
        total_chunks=total_chunks,
        page_type_distribution=dict(page_type_counts),
        section_type_distribution=dict(section_type_counts),
        quality_flag_distribution=dict(quality_flag_counts),
        canonical_pages=canonical_pages,
        non_canonical_pages=non_canonical_pages,
        weak_classifications=weak_classifications,
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_bytes(
        orjson.dumps(report.model_dump(mode="json"), option=orjson.OPT_INDENT_2)
    )
    return report


def run_semantic_normalization(
    chunks_path: Path = Path("datasets/normalized_chunks.jsonl"),
    documents_path: Path = Path("datasets/processed_documents.json"),
    report_path: Path = Path("reports/semantic_normalization_report.json"),
    min_paragraph_chars: int = 40,
    min_doc_count_for_reuse: int = 5,
    component_suppress_threshold: float = 0.5,
) -> SemanticNormalizationReport:
    """Phase 3: component suppression, hierarchy synthesis, mixed-topic detection,
    contamination flagging. Reads + writes back normalized_chunks.jsonl in-place."""
    if not chunks_path.exists():
        raise FileNotFoundError(f"normalized chunks file not found: {chunks_path}")

    chunks: list[dict] = [
        orjson.loads(line)
        for line in chunks_path.read_bytes().splitlines()
        if line.strip()
    ]
    if not chunks:
        raise ValueError(f"no chunks found in {chunks_path}")

    title_lookup = _build_document_title_lookup(documents_path)

    registry = build_registry(chunks, min_paragraph_chars=min_paragraph_chars)
    component_type_counts = classify_registry(registry, min_doc_count=min_doc_count_for_reuse)

    reusable_components = 0
    contaminated_chunks = 0
    mixed_topic_chunks = 0
    hierarchy_sections_extracted = 0
    generic_overview_replaced = 0
    semantic_sections_created = 0
    contamination_source_counts: Counter[str] = Counter()

    enriched: list[dict] = []
    for chunk in chunks:
        document_id = chunk.get("document_id", "")
        document_title = title_lookup.get(document_id, chunk.get("title", ""))

        section_path = normalize_section_path(
            page_type=chunk.get("page_type", ""),
            section_type=chunk.get("section_type", ""),
            document_title=document_title,
            metadata_headings=(chunk.get("metadata") or {}).get("headings") or [],
        )
        if section_path:
            hierarchy_sections_extracted += 1
            current_heading = (chunk.get("section_heading") or "").strip().lower()
            if current_heading in {"overview", "general", "introduction", ""}:
                generic_overview_replaced += 1
        if len(section_path) >= 2:
            semantic_sections_created += 1

        is_reusable, component_type, component_types_present = assess_chunk_components(
            chunk.get("text", ""),
            registry,
            min_paragraph_chars=min_paragraph_chars,
            suppress_threshold=component_suppress_threshold,
        )
        if is_reusable:
            reusable_components += 1

        is_mixed, dominant_topics = detect_mixed_topic(
            chunk.get("text", ""), min_paragraph_chars=min_paragraph_chars
        )
        if is_mixed:
            mixed_topic_chunks += 1

        is_contaminated, contamination_sources = detect_cross_domain_contamination(
            page_type=chunk.get("page_type", ""),
            component_types_in_chunk=component_types_present,
            mixed_topic=is_mixed,
            dominant_topics=dominant_topics,
        )
        if is_contaminated:
            contaminated_chunks += 1
        for source in contamination_sources:
            contamination_source_counts[source] += 1

        flags = list(chunk.get("quality_flags") or [])
        for flag, condition in (
            ("reusable_component", is_reusable),
            ("cross_domain_contamination", is_contaminated),
            ("mixed_topic", is_mixed),
        ):
            if condition and flag not in flags:
                flags.append(flag)

        new_priority = compute_retrieval_priority(
            page_type=chunk.get("page_type", ""),
            quality_flags=flags,
            page_type_confidence=chunk.get("page_type_confidence", 0.0),
            quality_score=chunk.get("quality_score", 0.0),
        )

        chunk["section_path"] = section_path
        chunk["is_reusable_component"] = is_reusable
        chunk["component_type"] = component_type
        chunk["mixed_topic"] = is_mixed
        chunk["dominant_topics"] = dominant_topics
        chunk["cross_domain_contamination"] = is_contaminated
        chunk["contamination_sources"] = contamination_sources
        chunk["quality_flags"] = flags
        chunk["retrieval_priority"] = new_priority
        chunk["embedding_eligible"] = is_embedding_eligible(chunk)
        enriched.append(chunk)

    tmp_path = chunks_path.with_suffix(chunks_path.suffix + ".tmp")
    with tmp_path.open("wb") as out_file:
        for chunk in enriched:
            out_file.write(orjson.dumps(chunk))
            out_file.write(b"\n")
    tmp_path.replace(chunks_path)

    report = SemanticNormalizationReport(
        total_chunks=len(enriched),
        reusable_components_detected=reusable_components,
        component_type_distribution=component_type_counts,
        contaminated_chunks=contaminated_chunks,
        contamination_source_distribution=dict(contamination_source_counts),
        mixed_topic_chunks=mixed_topic_chunks,
        hierarchy_sections_extracted=hierarchy_sections_extracted,
        generic_overview_sections_replaced=generic_overview_replaced,
        semantic_sections_created=semantic_sections_created,
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_bytes(
        orjson.dumps(report.model_dump(mode="json"), option=orjson.OPT_INDENT_2)
    )
    return report


def main() -> None:
    corpus_report = run_normalization()
    print(f"[corpus] Normalized {corpus_report.total_chunks} chunks")
    print(f"[corpus] Canonical pages: {corpus_report.canonical_pages} | Non-canonical: {corpus_report.non_canonical_pages}")
    print(f"[corpus] Weak classifications: {corpus_report.weak_classifications}")
    print(f"[corpus] Page types: {corpus_report.page_type_distribution}")

    semantic_report = run_semantic_normalization()
    print(f"\n[semantic] Reusable components: {semantic_report.reusable_components_detected}")
    print(f"[semantic] Contaminated chunks: {semantic_report.contaminated_chunks}")
    print(f"[semantic] Mixed-topic chunks: {semantic_report.mixed_topic_chunks}")
    print(f"[semantic] Hierarchy sections extracted: {semantic_report.hierarchy_sections_extracted}")
    print(f"[semantic] Generic 'Overview' sections replaced: {semantic_report.generic_overview_sections_replaced}")
    print(f"[semantic] Component types: {semantic_report.component_type_distribution}")


if __name__ == "__main__":
    main()
