from backend.conversation.memory import ConversationMemory


def test_get_creates_new_state() -> None:
    memory = ConversationMemory()
    state = memory.get("s1")
    assert state.session_id == "s1"
    assert state.turns == []


def test_append_user_turn_extracts_entities() -> None:
    memory = ConversationMemory()
    state = memory.append_user_turn("s1", "What is MCA eligibility?")
    assert state.active_entities.get("program") == "MCA"
    assert len(state.turns) == 1
    assert state.turns[0].role == "user"


def test_append_user_turn_preserves_prior_entities() -> None:
    memory = ConversationMemory()
    memory.append_user_turn("s1", "What is MCA eligibility?")
    state = memory.append_user_turn("s1", "What about hostel fees?")
    assert state.active_entities.get("program") == "MCA"


def test_append_assistant_turn_updates_intent_and_chunks() -> None:
    memory = ConversationMemory()
    memory.append_user_turn("s1", "MCA eligibility")
    state = memory.append_assistant_turn(
        "s1",
        "Eligibility is X.",
        citations=["c1", "c2"],
        intent="eligibility_query",
        routing_filters={"page_types": ["Admissions"]},
        used_chunks=["c1", "c2"],
    )
    assert state.last_intent == "eligibility_query"
    assert state.last_used_chunks == ["c1", "c2"]
    assert state.last_routing_filters == {"page_types": ["Admissions"]}


def test_memory_trims_window() -> None:
    memory = ConversationMemory(max_turns=4)
    for i in range(10):
        memory.append_user_turn("s1", f"q{i}")
    state = memory.get("s1")
    assert len(state.turns) == 4


def test_reset_clears_session() -> None:
    memory = ConversationMemory()
    memory.append_user_turn("s1", "MCA")
    assert "s1" in memory
    memory.reset("s1")
    assert "s1" not in memory
