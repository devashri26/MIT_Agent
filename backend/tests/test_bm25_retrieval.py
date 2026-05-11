from pathlib import Path

import orjson

from backend.retrieval.bm25_service import BM25RetrievalService


def _normalized_chunk(
    chunk_id: str,
    document_id: str,
    url: str,
    title: str,
    page_type: str,
    section_type: str,
    section_heading: str,
    text: str,
    retrieval_priority: float = 0.95,
    quality_flags: list[str] | None = None,
    is_reusable_component: bool = False,
    component_type: str | None = None,
) -> dict:
    return {
        "chunk_id": chunk_id,
        "document_id": document_id,
        "url": url,
        "canonical_url": url,
        "title": title,
        "department": None,
        "section_heading": section_heading,
        "section_path": [page_type, section_heading] if section_heading else [page_type],
        "chunk_index": 0,
        "text": text,
        "token_count": len(text.split()),
        "content_type": "GENERAL",
        "quality_score": 1.0,
        "chunk_hash": chunk_id,
        "page_type": page_type,
        "page_type_confidence": 0.95,
        "section_type": section_type,
        "retrieval_priority": retrieval_priority,
        "quality_flags": quality_flags or [],
        "is_canonical": True,
        "is_reusable_component": is_reusable_component,
        "component_type": component_type,
        "mixed_topic": False,
        "dominant_topics": [],
        "cross_domain_contamination": False,
        "contamination_sources": [],
        "metadata": {"headings": [section_heading], "quality_warnings": []},
    }


def _write_fixture(path: Path, chunks: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\n".join(orjson.dumps(c) for c in chunks))


def test_routed_search_prefers_admissions_for_eligibility_query(tmp_path: Path) -> None:
    path = tmp_path / "chunks.jsonl"
    _write_fixture(path, [
        _normalized_chunk(
            "c1", "doc1", "https://example.edu/mca",
            "MCA Admissions", "Admissions", "eligibility", "Eligibility",
            "MCA eligibility requires a relevant undergraduate degree and entrance examination score.",
            retrieval_priority=0.95,
        ),
        _normalized_chunk(
            "c2", "doc2", "https://example.edu/blog/post",
            "Blog post about MCA", "Blog", "general", "Article",
            "MCA eligibility tips and tricks discussed in this blog post.",
            retrieval_priority=0.4,
        ),
    ])

    response = BM25RetrievalService(path).search("What is MCA eligibility?", top_k=2)

    assert response.intent == "eligibility_query"
    assert response.results[0].chunk_id == "c1"
    assert response.results[0].page_type == "Admissions"
    assert response.results[0].explanation.metadata_boost == "Admissions"
    assert response.results[0].explanation.section_match == "eligibility"
    assert response.results[0].explanation.page_type_match is True


def test_filter_fallback_when_intent_eliminates_all(tmp_path: Path) -> None:
    path = tmp_path / "chunks.jsonl"
    _write_fixture(path, [
        _normalized_chunk(
            "c1", "doc1", "https://example.edu/blog/x",
            "Hostel Stories", "Blog", "general", "Hostel life",
            "Living in hostel was an unforgettable experience.",
            retrieval_priority=0.4,
        ),
    ])

    response = BM25RetrievalService(path).search("hostel facilities", top_k=1)

    assert response.intent == "hostel_query"
    assert response.filter_fallback_used is True
    assert len(response.results) == 1


def test_explanation_includes_matched_query_terms(tmp_path: Path) -> None:
    path = tmp_path / "chunks.jsonl"
    _write_fixture(path, [
        _normalized_chunk(
            "c1", "doc1", "https://example.edu/admissions",
            "Admissions", "Admissions", "admissions", "Overview",
            "Admissions process and eligibility for all programs.",
        ),
    ])
    response = BM25RetrievalService(path).search("eligibility", top_k=1)
    assert "eligibility" in response.results[0].explanation.matched_terms


def test_reusable_components_excluded_by_default(tmp_path: Path) -> None:
    path = tmp_path / "chunks.jsonl"
    _write_fixture(path, [
        _normalized_chunk(
            "c1", "doc1", "https://example.edu/admissions",
            "Admissions", "Admissions", "admissions", "Overview",
            "MCA admissions eligibility criteria for all candidates.",
            is_reusable_component=True,
            component_type="admissions_cta",
        ),
        _normalized_chunk(
            "c2", "doc2", "https://example.edu/mca",
            "MCA Admissions", "Admissions", "eligibility", "Eligibility",
            "MCA eligibility requires a relevant undergraduate degree and entrance exam.",
        ),
    ])
    response = BM25RetrievalService(path).search("MCA eligibility", top_k=5)
    assert response.components_excluded == 1
    assert all(r.chunk_id != "c1" for r in response.results)


def test_include_components_overrides_default(tmp_path: Path) -> None:
    path = tmp_path / "chunks.jsonl"
    _write_fixture(path, [
        _normalized_chunk(
            "c1", "doc1", "https://example.edu/admissions",
            "Admissions", "Admissions", "admissions", "Overview",
            "MCA admissions eligibility criteria for all candidates.",
            is_reusable_component=True,
            component_type="admissions_cta",
        ),
    ])
    response = BM25RetrievalService(path).search(
        "MCA eligibility", top_k=5, include_components=True
    )
    assert response.components_excluded == 0
    assert len(response.results) == 1
