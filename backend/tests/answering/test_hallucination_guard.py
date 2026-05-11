from backend.answering.hallucination_guard import _parse_judge_output, validate_grounded_answer
from backend.llm.mock_provider import MockLLMProvider


def test_parse_judge_output_clean_json() -> None:
    risk, claims, err = _parse_judge_output('{"hallucination_risk": 0.3, "unsupported_claims": ["x"]}')
    assert risk == 0.3
    assert claims == ["x"]
    assert err is None


def test_parse_judge_output_json_in_prose() -> None:
    text = "Here is my answer: {\"hallucination_risk\": 0.4, \"unsupported_claims\": []} done."
    risk, claims, err = _parse_judge_output(text)
    assert risk == 0.4
    assert claims == []
    assert err is None


def test_parse_judge_output_garbage() -> None:
    risk, claims, err = _parse_judge_output("not json at all")
    assert risk == 0.0
    assert claims == []
    assert err == "no_json_object"


def test_parse_judge_output_empty() -> None:
    risk, claims, err = _parse_judge_output("")
    assert err == "empty_response"


def test_validate_with_mock_judge_returns_safe(grounded_context_two_blocks) -> None:
    judge = MockLLMProvider(canned_response='{"hallucination_risk": 0.1, "unsupported_claims": []}')
    check = validate_grounded_answer(
        provider=judge,
        answer="Eligibility is 50 percent [1].",
        grounded_context=grounded_context_two_blocks,
    )
    assert check.safe_to_return is True
    assert check.hallucination_risk == 0.1
    assert check.judge_used is True
    assert check.judge_error is None


def test_validate_flags_unsafe_when_risk_high(grounded_context_two_blocks) -> None:
    judge = MockLLMProvider(canned_response='{"hallucination_risk": 0.8, "unsupported_claims": ["Faculty X invented Y"]}')
    check = validate_grounded_answer(
        provider=judge,
        answer="Faculty X invented Y [1].",
        grounded_context=grounded_context_two_blocks,
    )
    assert check.safe_to_return is False
    assert check.hallucination_risk == 0.8
    assert "Faculty X invented Y" in check.unsupported_claims


def test_validate_handles_garbage_judge_output(grounded_context_two_blocks) -> None:
    judge = MockLLMProvider(canned_response="this is not json")
    check = validate_grounded_answer(
        provider=judge,
        answer="Some answer [1].",
        grounded_context=grounded_context_two_blocks,
    )
    assert check.judge_used is True
    assert check.judge_error is not None
    assert check.hallucination_risk == 0.0
