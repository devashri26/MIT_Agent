from __future__ import annotations

from backend.conversation.validators import ConversationState


def augment_query_with_state(query: str, state: ConversationState) -> str:
    """If the rewriter didn't produce a fully-specified query, append known active
    entities as a hint so retrieval routing has something to bind to. Conservative — only
    appends when the query lacks the entity values."""
    if not state.active_entities:
        return query
    lower = query.lower()
    additions: list[str] = []
    for key, value in state.active_entities.items():
        if value and value.lower() not in lower:
            additions.append(value)
    if not additions:
        return query
    return f"{query} ({', '.join(additions)})"
