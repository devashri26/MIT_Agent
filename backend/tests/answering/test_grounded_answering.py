from backend.answering.grounded_answering import GroundedAnsweringService
from backend.answering.models.answer import ABSTENTION_TEXT
from backend.llm.mock_provider import MockLLMProvider


def test_answers_with_citations(grounded_context_two_blocks) -> None:
    provider = MockLLMProvider(canned_response="Eligibility is 50% [1]. Fees are 1.2 lakh [2].")
    judge = MockLLMProvider(canned_response='{"hallucination_risk": 0.05, "unsupported_claims": []}')
    service = GroundedAnsweringService(provider=provider, judge_provider=judge)
    result = service.answer(grounded_context_two_blocks.query, grounded_context_two_blocks)
    assert result.abstained is False
    assert len(result.citations) == 2
    assert result.confidence.answer_confidence > 0
    assert result.confidence.citation_coverage == 1.0
    assert result.used_chunks == ["c1", "c2"]


def test_abstains_when_no_context(grounded_context_empty) -> None:
    provider = MockLLMProvider(canned_response="should not be reached")
    service = GroundedAnsweringService(provider=provider, run_judge=False)
    result = service.answer(grounded_context_empty.query, grounded_context_empty)
    assert result.abstained is True
    assert result.abstention_reason == "no_context_blocks"
    assert result.answer == ABSTENTION_TEXT


def test_abstains_when_judge_flags_high_risk(grounded_context_two_blocks) -> None:
    provider = MockLLMProvider(canned_response="Made-up facts [1].")
    judge = MockLLMProvider(canned_response='{"hallucination_risk": 0.9, "unsupported_claims": ["fact"]}')
    service = GroundedAnsweringService(provider=provider, judge_provider=judge)
    result = service.answer(grounded_context_two_blocks.query, grounded_context_two_blocks)
    assert result.abstained is True
    assert result.abstention_reason.startswith("high_hallucination_risk")


def test_passes_model_self_abstention_through(grounded_context_two_blocks) -> None:
    provider = MockLLMProvider(canned_response=ABSTENTION_TEXT)
    service = GroundedAnsweringService(provider=provider, run_judge=False)
    result = service.answer(grounded_context_two_blocks.query, grounded_context_two_blocks)
    assert result.abstained is True
    assert result.abstention_reason == "model_self_abstained"


def test_warns_on_missing_citations(grounded_context_two_blocks) -> None:
    provider = MockLLMProvider(canned_response="Plain prose, no citations.")
    judge = MockLLMProvider(canned_response='{"hallucination_risk": 0.05, "unsupported_claims": []}')
    service = GroundedAnsweringService(provider=provider, judge_provider=judge)
    result = service.answer(grounded_context_two_blocks.query, grounded_context_two_blocks)
    assert result.abstained is False
    assert any("no_citations_in_answer" in w for w in result.grounding_warnings)
