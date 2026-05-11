from backend.conversation.validators import ConversationTurn


DEFAULT_MAX_TURNS = 6


def trim_to_window(
    turns: list[ConversationTurn],
    max_turns: int = DEFAULT_MAX_TURNS,
) -> list[ConversationTurn]:
    """Keep only the most recent N turns. Older turns drop off; entities are preserved on
    ConversationState so dropped turns don't fully lose their context."""
    if max_turns <= 0:
        return []
    return turns[-max_turns:]
