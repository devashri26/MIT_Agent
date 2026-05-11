from backend.context.context_builder import build_grounded_context
from backend.reranking.validators import RerankedChunk
from backend.retrieval.retrieval_debugger import RetrievalExplanation


def _reranked(chunk_id: str, text: str, section_type: str, final_relevance: float = 0.8) -> RerankedChunk:
    return RerankedChunk(
        rank=1,
        score=final_relevance,
        chunk_id=chunk_id,
        document_id=f"doc_{chunk_id}",
        url=f"https://example.edu/{chunk_id}",
        canonical_url=f"https://example.edu/{chunk_id}",
        title=text[:30],
        page_type="Admissions",
        section_type=section_type,
        section_heading="Overview",
        section_path=["Admissions", section_type.capitalize()],
        token_count=len(text.split()),
        content_type="GENERAL",
        quality_score=1.0,
        retrieval_priority=0.9,
        text=text,
        explanation=RetrievalExplanation(intent="eligibility_query"),
        rerank_score=final_relevance,
        answerability_score=0.6,
        final_relevance=final_relevance,
    )


def test_builder_produces_blocks_with_citations() -> None:
    reranked = [
        _reranked("a", "MCA eligibility minimum 50 percent in graduation degree", "eligibility"),
        _reranked("b", "MCA fee structure tuition charges per semester", "fees"),
    ]
    ctx = build_grounded_context("MCA eligibility and fees", "eligibility_query", reranked)
    assert len(ctx.context_blocks) == 2
    assert ctx.context_blocks[0].citation.chunk_id == "a"
    assert ctx.context_blocks[0].citation.source_url.startswith("https://")
    assert ctx.distinct_section_types == 2
    assert ctx.grounding_confidence > 0
    assert "Cite sources using [1]" in ctx.prompt


def test_builder_respects_token_budget() -> None:
    big_text = ("very long paragraph " * 200)
    reranked = [
        _reranked("a", big_text, "eligibility"),
        _reranked("b", big_text, "fees"),
        _reranked("c", "MCA admissions process", "admissions"),
    ]
    ctx = build_grounded_context("MCA", "general_query", reranked, token_budget=300)
    assert ctx.total_tokens <= 300
    # Some should be dropped
    assert len(ctx.dropped_blocks) >= 1


def test_builder_empty_input_emits_warning() -> None:
    ctx = build_grounded_context("query", "general_query", [])
    assert ctx.context_blocks == []
    assert "no_reranked_candidates" in ctx.grounding_warnings


def test_builder_records_dropped_duplicates() -> None:
    reranked = [
        _reranked("a", "MCA eligibility minimum qualification entrance examination", "eligibility"),
        _reranked("b", "MCA eligibility minimum qualification entrance examination", "eligibility"),
    ]
    ctx = build_grounded_context("MCA eligibility", "eligibility_query", reranked)
    chunk_ids = [b.chunk_id for b in ctx.context_blocks]
    dropped_ids = [d.chunk_id for d in ctx.dropped_blocks]
    assert "a" in chunk_ids
    assert "b" in dropped_ids
