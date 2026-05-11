from backend.answering.abstention import should_abstain
from backend.answering.models.answer import HallucinationCheck


def test_abstain_on_empty_context(grounded_context_empty) -> None:
    abstain, reason = should_abstain(grounded_context_empty)
    assert abstain is True
    assert reason == "no_context_blocks"


def test_no_abstain_on_strong_context(grounded_context_two_blocks) -> None:
    abstain, reason = should_abstain(grounded_context_two_blocks)
    assert abstain is False
    assert reason is None


def test_abstain_when_grounding_confidence_too_low(grounded_context_two_blocks) -> None:
    grounded_context_two_blocks.grounding_confidence = 0.1
    abstain, reason = should_abstain(grounded_context_two_blocks)
    assert abstain is True
    assert reason.startswith("low_grounding_confidence")


def test_abstain_on_high_hallucination_risk(grounded_context_two_blocks) -> None:
    check = HallucinationCheck(hallucination_risk=0.7, safe_to_return=False, judge_used=True)
    abstain, reason = should_abstain(grounded_context_two_blocks, hallucination=check)
    assert abstain is True
    assert reason.startswith("high_hallucination_risk")
