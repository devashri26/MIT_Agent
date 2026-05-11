from backend.context.grounding import compute_grounding_confidence, validate_grounded_context
from backend.context.validators import Citation, ContextBlock


def _block(chunk_id: str, relevance: float, section_type: str = "eligibility") -> ContextBlock:
    return ContextBlock(
        chunk_id=chunk_id,
        document_id=f"doc_{chunk_id}",
        text="x",
        citation=Citation(chunk_id=chunk_id, source_url="https://x", title="t"),
        source_url="https://x",
        title="t",
        page_type="Admissions",
        section_type=section_type,
        final_relevance=relevance,
        token_count=10,
    )


def test_grounding_confidence_is_mean_of_relevance() -> None:
    blocks = [_block("a", 0.8), _block("b", 0.6)]
    assert compute_grounding_confidence(blocks) == 0.7


def test_grounding_confidence_empty() -> None:
    assert compute_grounding_confidence([]) == 0.0


def test_validate_no_blocks_warns() -> None:
    confidence, warnings = validate_grounded_context([])
    assert confidence == 0.0
    assert "no_blocks" in warnings


def test_validate_low_confidence_warns() -> None:
    blocks = [_block("a", 0.1), _block("b", 0.2)]
    confidence, warnings = validate_grounded_context(blocks, min_confidence=0.5)
    assert any(w.startswith("low_confidence") for w in warnings)


def test_validate_insufficient_blocks_warns() -> None:
    confidence, warnings = validate_grounded_context([_block("a", 0.9)], min_blocks=2)
    assert any(w.startswith("insufficient_blocks") for w in warnings)


def test_validate_low_topical_diversity_warns() -> None:
    blocks = [
        _block("a", 0.9, section_type="eligibility"),
        _block("b", 0.8, section_type="eligibility"),
        _block("c", 0.7, section_type="eligibility"),
    ]
    confidence, warnings = validate_grounded_context(blocks)
    assert any(w.startswith("low_topical_diversity") for w in warnings)


def test_validate_clean_context() -> None:
    blocks = [
        _block("a", 0.8, section_type="eligibility"),
        _block("b", 0.7, section_type="fees"),
        _block("c", 0.75, section_type="curriculum"),
    ]
    confidence, warnings = validate_grounded_context(blocks)
    assert confidence > 0.5
    assert warnings == []
