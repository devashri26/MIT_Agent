from backend.context.token_budget import count_tokens, fit_to_budget
from backend.context.validators import Citation, ContextBlock


def _block(chunk_id: str, text: str, tokens: int) -> ContextBlock:
    return ContextBlock(
        chunk_id=chunk_id,
        document_id=f"doc_{chunk_id}",
        text=text,
        citation=Citation(chunk_id=chunk_id, source_url="https://x", title="t"),
        source_url="https://x",
        title="t",
        page_type="Admissions",
        section_type="eligibility",
        token_count=tokens,
    )


def test_count_tokens_nonzero_for_text() -> None:
    assert count_tokens("hello world") > 0


def test_count_tokens_empty() -> None:
    assert count_tokens("") == 0


def test_fit_to_budget_keeps_within_limit() -> None:
    blocks = [_block("a", "x", 500), _block("b", "y", 800), _block("c", "z", 800)]
    kept, dropped = fit_to_budget(blocks, max_tokens=1500)
    assert [b.chunk_id for b in kept] == ["a", "b"]
    assert len(dropped) == 1
    assert dropped[0].chunk_id == "c"


def test_fit_to_budget_drops_oversize_first_block() -> None:
    blocks = [_block("a", "x", 5000), _block("b", "y", 100)]
    kept, dropped = fit_to_budget(blocks, max_tokens=1000)
    assert [b.chunk_id for b in kept] == ["b"]
    assert dropped[0].chunk_id == "a"


def test_fit_to_budget_empty_input() -> None:
    kept, dropped = fit_to_budget([], max_tokens=1000)
    assert kept == [] and dropped == []
