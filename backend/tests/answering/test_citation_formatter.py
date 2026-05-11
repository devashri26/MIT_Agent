from backend.answering.citation_formatter import build_citations, extract_cited_indices
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


def test_extract_cited_indices_ordered_and_deduped() -> None:
    text = "Eligibility [1] requires a degree [2]. Also see [1] again."
    assert extract_cited_indices(text) == [1, 2]


def test_extract_cited_indices_no_match() -> None:
    assert extract_cited_indices("plain text without citations") == []


def test_build_citations_maps_indices_to_blocks() -> None:
    blocks = [_block("c1"), _block("c2")]
    citations = build_citations("See [1] and [2].", blocks)
    assert [c.index for c in citations] == [1, 2]
    assert [c.chunk_id for c in citations] == ["c1", "c2"]


def test_build_citations_skips_out_of_range() -> None:
    blocks = [_block("c1")]
    citations = build_citations("Bad ref [5].", blocks)
    assert citations == []


def test_build_citations_empty_answer() -> None:
    blocks = [_block("c1")]
    assert build_citations("", blocks) == []
