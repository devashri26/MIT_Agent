from backend.reranking.rerank_service import RerankService
from backend.retrieval.models.search import RetrievedChunk
from backend.retrieval.retrieval_debugger import RetrievalExplanation


def _chunk(
    chunk_id: str,
    text: str,
    *,
    page_type: str = "Admissions",
    section_type: str = "eligibility",
    document_id: str | None = None,
    rank: int = 1,
) -> RetrievedChunk:
    return RetrievedChunk(
        rank=rank,
        score=0.5,
        chunk_id=chunk_id,
        document_id=document_id or f"doc_{chunk_id}",
        url=f"https://example.edu/{chunk_id}",
        canonical_url=f"https://example.edu/{chunk_id}",
        title=text[:30],
        page_type=page_type,
        section_type=section_type,
        section_heading="Overview",
        section_path=[page_type, section_type],
        token_count=len(text.split()),
        content_type="GENERAL",
        quality_score=1.0,
        retrieval_priority=0.9,
        text=text,
        explanation=RetrievalExplanation(intent="eligibility_query"),
    )


def test_rerank_orders_by_query_overlap(fake_reranker) -> None:
    service = RerankService(model=fake_reranker)
    candidates = [
        _chunk("a", "hostel facilities rooms mess accommodation"),
        _chunk("b", "MCA eligibility minimum qualification entrance"),
        _chunk("c", "blog generic content about campus life"),
    ]
    kept, rejected = service.rerank("MCA eligibility minimum qualification", candidates, top_k=3)
    assert kept[0].chunk_id == "b"
    assert kept[0].rerank_score > 0
    assert kept[0].final_relevance > 0


def test_rerank_marks_duplicates_in_rejected(fake_reranker) -> None:
    service = RerankService(model=fake_reranker)
    candidates = [
        _chunk("a", "MCA eligibility minimum qualification entrance exam"),
        _chunk("b", "MCA eligibility minimum qualification entrance exam"),
        _chunk("c", "hostel facilities accommodation rooms"),
    ]
    kept, rejected = service.rerank("MCA eligibility entrance", candidates, top_k=5)
    kept_ids = [k.chunk_id for k in kept]
    rejected_ids = [r.chunk_id for r in rejected]
    assert "a" in kept_ids
    assert "b" in rejected_ids
    duplicate = next(r for r in rejected if r.chunk_id == "b")
    assert duplicate.rejection_reason is not None
    assert duplicate.rejection_reason.startswith("duplicate_of")


def test_rerank_diversity_caps_per_section(fake_reranker) -> None:
    service = RerankService(model=fake_reranker)
    candidates = [
        _chunk("e1", "Candidates require bachelor degree with minimum aggregate marks defined"),
        _chunk("e2", "Entrance exam scores from PERA CET MAH MCA CET accepted"),
        _chunk("e3", "Reservation criteria for various government categories outlined separately here"),
        _chunk("e4", "Age limit twenty eight years unreserved category applying mca"),
        _chunk("fees", "MCA fee structure tuition charges details", section_type="fees"),
    ]
    kept, rejected = service.rerank(
        "MCA eligibility criteria minimum candidates",
        candidates,
        top_k=5,
        max_per_section_type=2,
    )
    kept_section_types = [k.section_type for k in kept]
    assert kept_section_types.count("eligibility") <= 2
    eligibility_rejects = [r for r in rejected if r.section_type == "eligibility"]
    assert any(
        r.rejection_reason and "section_type_saturated" in r.rejection_reason
        for r in eligibility_rejects
    )


def test_rerank_assigns_sequential_ranks(fake_reranker) -> None:
    service = RerankService(model=fake_reranker)
    candidates = [
        _chunk("a", "MCA eligibility minimum qualification"),
        _chunk("b", "MCA fees structure", section_type="fees"),
        _chunk("c", "MCA admissions process timeline", section_type="admissions"),
    ]
    kept, _ = service.rerank("MCA eligibility fees admissions", candidates, top_k=3)
    assert [k.rank for k in kept] == [1, 2, 3]


def test_rerank_empty_input(fake_reranker) -> None:
    service = RerankService(model=fake_reranker)
    kept, rejected = service.rerank("query", [], top_k=5)
    assert kept == []
    assert rejected == []
