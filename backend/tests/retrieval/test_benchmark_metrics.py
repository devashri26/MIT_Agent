from backend.retrieval.benchmark_metrics import compute_metrics
from backend.retrieval.models.search import RetrievedChunk, SearchResponse
from backend.retrieval.retrieval_debugger import RetrievalExplanation


def _make_chunk(rank: int, page_type: str, chunk_id: str) -> RetrievedChunk:
    return RetrievedChunk(
        rank=rank,
        score=1.0 / rank,
        chunk_id=chunk_id,
        document_id=chunk_id,
        url="https://example.edu",
        canonical_url="https://example.edu",
        title="t",
        page_type=page_type,
        section_type="overview",
        section_heading="h",
        token_count=100,
        content_type="GENERAL",
        quality_score=1.0,
        retrieval_priority=0.9,
        quality_flags=[],
        text="t",
        explanation=RetrievalExplanation(intent="eligibility_query"),
    )


def test_metrics_perfect_routing() -> None:
    response = SearchResponse(
        query="What is MCA eligibility?",
        top_k=10,
        intent="eligibility_query",
        allowed_page_types=["Admissions", "Programs"],
        allowed_section_types=["eligibility", "admissions"],
        results=[
            _make_chunk(1, "Admissions", "c1"),
            _make_chunk(2, "Programs", "c2"),
            _make_chunk(3, "Admissions", "c3"),
        ],
    )
    corpus = [
        {"page_type": "Admissions"} for _ in range(10)
    ] + [{"page_type": "Programs"} for _ in range(20)] + [{"page_type": "Blog"} for _ in range(70)]

    metrics = compute_metrics([response], corpus)

    assert metrics["overall"]["hit_rate"] == 1.0
    assert metrics["overall"]["mrr"] == 1.0
    assert metrics["overall"]["precision@3"] == 1.0
    assert metrics["per_query"][0]["matches_top10"] == 3


def test_metrics_no_routing_match() -> None:
    response = SearchResponse(
        query="What is MCA eligibility?",
        top_k=10,
        intent="eligibility_query",
        allowed_page_types=["Admissions"],
        allowed_section_types=[],
        results=[
            _make_chunk(1, "Blog", "c1"),
            _make_chunk(2, "Blog", "c2"),
        ],
    )
    corpus = [{"page_type": "Admissions"} for _ in range(10)]

    metrics = compute_metrics([response], corpus)
    assert metrics["overall"]["hit_rate"] == 0.0
    assert metrics["overall"]["mrr"] == 0.0
    assert metrics["overall"]["precision@3"] == 0.0


def test_per_intent_breakdown() -> None:
    r1 = SearchResponse(
        query="q1", top_k=10, intent="eligibility_query",
        allowed_page_types=["Admissions"], allowed_section_types=[],
        results=[_make_chunk(1, "Admissions", "c1")],
    )
    r2 = SearchResponse(
        query="q2", top_k=10, intent="placement_query",
        allowed_page_types=["Placements"], allowed_section_types=[],
        results=[_make_chunk(1, "Blog", "c2")],
    )
    corpus = [{"page_type": "Admissions"}, {"page_type": "Placements"}]
    metrics = compute_metrics([r1, r2], corpus)
    assert metrics["per_intent"]["eligibility_query"]["hit_rate"] == 1.0
    assert metrics["per_intent"]["placement_query"]["hit_rate"] == 0.0
