from backend.conversation.query_rewriter import is_followup_query, rewrite_query
from backend.conversation.validators import ConversationState, ConversationTurn
from backend.llm.mock_provider import MockLLMProvider


def _state_with_turn(content: str = "What is MCA eligibility?", program: str = "MCA") -> ConversationState:
    state = ConversationState(session_id="s1")
    state.turns.append(ConversationTurn(role="user", content=content))
    state.active_entities = {"program": program}
    state.last_intent = "eligibility_query"
    return state


def test_is_followup_short_query_with_marker() -> None:
    state = _state_with_turn()
    assert is_followup_query("And hostel fees?", state) is True


def test_short_standalone_query_is_not_followup() -> None:
    """Regression: 'who is dr sunita barve' (5 words, no followup marker) was being
    treated as a followup and rewritten to 'who is' by the LLM, destroying retrieval."""
    state = _state_with_turn()
    assert is_followup_query("who is dr sunita barve", state) is False
    assert is_followup_query("MCA admission deadline", state) is False
    assert is_followup_query("fees structure", state) is False


def test_short_followup_with_marker_still_caught() -> None:
    state = _state_with_turn()
    assert is_followup_query("and MTech?", state) is True
    assert is_followup_query("what about that?", state) is True


def test_is_followup_with_marker() -> None:
    state = _state_with_turn()
    assert is_followup_query("What about placements for the same program?", state) is True


def test_not_followup_when_long_and_specific() -> None:
    state = _state_with_turn()
    assert is_followup_query(
        "Please tell me the eligibility for MCA admissions in detail",
        state,
    ) is False


def test_not_followup_when_no_history() -> None:
    state = ConversationState(session_id="s1")
    assert is_followup_query("What about hostel?", state) is False


def test_rewrite_skips_when_not_followup() -> None:
    state = ConversationState(session_id="s1")
    provider = MockLLMProvider(canned_response="REWRITTEN")
    result = rewrite_query(provider, "Standalone question with enough detail", state)
    assert result == "Standalone question with enough detail"


def test_rewrite_calls_llm_for_followup() -> None:
    state = _state_with_turn()
    provider = MockLLMProvider(canned_response="What are the hostel fees for MCA?")
    result = rewrite_query(provider, "What about hostel fees?", state)
    assert result == "What are the hostel fees for MCA?"


def test_rewrite_falls_back_to_original_on_empty_response() -> None:
    state = _state_with_turn()
    provider = MockLLMProvider(canned_response="")
    result = rewrite_query(provider, "What about hostel fees?", state)
    assert result == "What about hostel fees?"


def test_rewrite_falls_back_when_truncated() -> None:
    """If the rewriter returns something noticeably shorter than the original (often a
    truncated 'who is' from a confused model), fall back to the original."""
    state = _state_with_turn()
    provider = MockLLMProvider(canned_response="who is")
    result = rewrite_query(provider, "and dr sunita barve in detail please", state)
    assert result == "and dr sunita barve in detail please"
