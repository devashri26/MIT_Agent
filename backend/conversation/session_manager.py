from __future__ import annotations

import uuid


def new_session_id() -> str:
    return str(uuid.uuid4())


def ensure_session_id(session_id: str | None) -> str:
    return session_id or new_session_id()
