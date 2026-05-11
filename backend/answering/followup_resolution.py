from __future__ import annotations

from backend.answering.conversational_context import augment_query_with_state
from backend.conversation.query_rewriter import is_followup_query, rewrite_query
from backend.conversation.validators import ConversationState
from backend.llm.provider_interface import BaseLLMProvider


def resolve_followup_query(
    provider: BaseLLMProvider,
    query: str,
    state: ConversationState,
    model: str = "",
) -> tuple[str, bool]:
    """Resolve a possibly-vague followup into a standalone query.

    Returns (resolved_query, was_rewritten). The augmentation pass adds active-entity
    hints when the LLM rewrite didn't pick them up, so retrieval routing has the maximum
    signal available."""
    is_followup = is_followup_query(query, state)
    rewritten = rewrite_query(provider=provider, query=query, state=state, model=model)
    augmented = augment_query_with_state(rewritten, state)
    return augmented, (is_followup and augmented != query)
