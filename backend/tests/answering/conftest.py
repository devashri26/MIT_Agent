import pytest

from backend.context.validators import Citation, ContextBlock, GroundedContext


def _block(chunk_id: str, text: str, section_type: str = "eligibility", page_type: str = "Admissions") -> ContextBlock:
    return ContextBlock(
        chunk_id=chunk_id,
        document_id=f"doc_{chunk_id}",
        text=text,
        citation=Citation(chunk_id=chunk_id, source_url=f"https://example.edu/{chunk_id}", title=text[:30]),
        source_url=f"https://example.edu/{chunk_id}",
        title=text[:30],
        page_type=page_type,
        section_type=section_type,
        section_path=[page_type, section_type.capitalize()],
        rerank_score=0.7,
        answerability_score=0.6,
        final_relevance=0.65,
        token_count=len(text.split()),
    )


@pytest.fixture
def grounded_context_two_blocks() -> GroundedContext:
    blocks = [
        _block("c1", "MCA eligibility minimum 50 percent in graduation."),
        _block("c2", "MCA fee structure tuition charges 1.2 lakh per year.", section_type="fees"),
    ]
    return GroundedContext(
        query="What is MCA eligibility and fees?",
        intent="eligibility_query",
        context_blocks=blocks,
        grounding_confidence=0.65,
        grounding_warnings=[],
        total_tokens=200,
        token_budget=2000,
        distinct_section_types=2,
        distinct_documents=2,
        prompt="Question: ...\n\n[1] ...\n[2] ...\n",
    )


@pytest.fixture
def grounded_context_empty() -> GroundedContext:
    return GroundedContext(
        query="random unrelated stuff",
        intent="general_query",
        context_blocks=[],
        grounding_confidence=0.0,
        grounding_warnings=["no_blocks"],
        token_budget=2000,
    )
