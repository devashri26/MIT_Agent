from __future__ import annotations

from backend.conversation.context_window import DEFAULT_MAX_TURNS, trim_to_window
from backend.conversation.retrieval_state import extract_entities
from backend.conversation.validators import ConversationState, ConversationTurn


class ConversationMemory:
    """In-memory session store: session_id → ConversationState. Persistence is out of
    scope for this phase; restart drops history."""

    def __init__(self, max_turns: int = DEFAULT_MAX_TURNS) -> None:
        self._store: dict[str, ConversationState] = {}
        self.max_turns = max_turns

    def get(self, session_id: str) -> ConversationState:
        if session_id not in self._store:
            self._store[session_id] = ConversationState(session_id=session_id)
        return self._store[session_id]

    def append_user_turn(self, session_id: str, content: str) -> ConversationState:
        state = self.get(session_id)
        state.turns.append(ConversationTurn(role="user", content=content))
        state.turns = trim_to_window(state.turns, self.max_turns)
        # Merge entities from this turn (don't overwrite previously-active program/department
        # unless the new query explicitly sets them).
        new_entities = extract_entities(content)
        for key, value in new_entities.items():
            state.active_entities[key] = value
        return state

    def append_assistant_turn(
        self,
        session_id: str,
        content: str,
        citations: list[str] | None = None,
        rewritten_query: str | None = None,
        intent: str | None = None,
        routing_filters: dict[str, list[str]] | None = None,
        used_chunks: list[str] | None = None,
    ) -> ConversationState:
        state = self.get(session_id)
        state.turns.append(
            ConversationTurn(
                role="assistant",
                content=content,
                citations=citations or [],
                rewritten_query=rewritten_query,
            )
        )
        state.turns = trim_to_window(state.turns, self.max_turns)
        if intent is not None:
            state.last_intent = intent
        if routing_filters is not None:
            state.last_routing_filters = routing_filters
        if used_chunks is not None:
            state.last_used_chunks = used_chunks
        return state

    def reset(self, session_id: str) -> None:
        self._store.pop(session_id, None)

    def __contains__(self, session_id: str) -> bool:
        return session_id in self._store
