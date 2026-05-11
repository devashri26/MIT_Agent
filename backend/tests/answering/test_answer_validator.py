from backend.answering.answer_validator import validate_answer
from backend.context.validators import Citation, ContextBlock


def _block(chunk_id: str) -> ContextBlock:
    return ContextBlock(
        chunk_id=chunk_id,
        document_id=f"doc_{chunk_id}",
        text="text",
        citation=Citation(chunk_id=chunk_id, source_url=f"https://example.edu/{chunk_id}", title=chunk_id),
        source_url=f"https://example.edu/{chunk_id}",
        title=chunk_id,
        page_type="Admissions",
        section_type="eligibility",
        token_count=10,
    )


def test_validate_coverage_when_all_blocks_cited() -> None:
    blocks = [_block("c1"), _block("c2")]
    coverage, warnings = validate_answer("See [1] and [2].", blocks)
    assert coverage == 1.0
    assert warnings == []


def test_validate_partial_coverage() -> None:
    blocks = [_block("c1"), _block("c2"), _block("c3")]
    coverage, warnings = validate_answer("Only [1].", blocks)
    assert coverage == round(1 / 3, 4)
    assert warnings == []


def test_validate_no_citation_warning() -> None:
    blocks = [_block("c1")]
    coverage, warnings = validate_answer("Plain answer.", blocks)
    assert coverage == 0.0
    assert "no_citations_in_answer" in warnings


def test_validate_out_of_range_citation_warning() -> None:
    blocks = [_block("c1")]
    coverage, warnings = validate_answer("Bad ref [5].", blocks)
    assert any(w.startswith("out_of_range_citations") for w in warnings)


def test_validate_empty_context() -> None:
    coverage, warnings = validate_answer("anything", [])
    assert coverage == 0.0
    assert "no_context_blocks" in warnings
