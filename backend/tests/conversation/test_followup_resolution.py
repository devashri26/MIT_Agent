from backend.answering.followup_resolution import resolve_followup_query
from backend.conversation.validators import ConversationState, ConversationTurn
from backend.llm.mock_provider import MockLLMProvider


def test_resolve_with_rewrite_and_entity_augmentation() -> None:
    state = ConversationState(session_id="s1")
    state.turns.append(ConversationTurn(role="user", content="MCA eligibility"))
    state.active_entities = {"program": "MCA"}
    provider = MockLLMProvider(canned_response="What are the placements")
    resolved, was_rewritten = resolve_followup_query(provider, "What about placements?", state)
    assert "MCA" in resolved
    assert was_rewritten is True


def test_standalone_query_passes_through() -> None:
    state = ConversationState(session_id="s1")
    provider = MockLLMProvider(canned_response="should not be used")
    resolved, was_rewritten = resolve_followup_query(
        provider,
        "Please give me the detailed eligibility criteria for the MCA admissions process",
        state,
    )
    assert was_rewritten is False


def test_augmentation_only_when_entity_missing() -> None:
    state = ConversationState(session_id="s1")
    state.turns.append(ConversationTurn(role="user", content="BTech curriculum"))
    state.active_entities = {"program": "BTech"}
    provider = MockLLMProvider(canned_response="BTech curriculum question")
    resolved, _ = resolve_followup_query(provider, "and that?", state)
    # Entity already present in rewritten — no duplicate
    assert resolved.count("BTech") == 1
