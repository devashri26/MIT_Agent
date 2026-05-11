from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field


class ConversationTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    citations: list[str] = Field(default_factory=list)
    rewritten_query: str | None = None


class ConversationState(BaseModel):
    session_id: str
    turns: list[ConversationTurn] = Field(default_factory=list)
    active_entities: dict[str, str] = Field(default_factory=dict)
    last_intent: str | None = None
    last_routing_filters: dict[str, list[str]] = Field(default_factory=dict)
    last_used_chunks: list[str] = Field(default_factory=list)
