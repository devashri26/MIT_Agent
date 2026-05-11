from __future__ import annotations

import tiktoken

from backend.context.validators import ContextBlock, DroppedBlock


_ENCODER = None


def _encoder() -> tiktoken.Encoding:
    global _ENCODER
    if _ENCODER is None:
        _ENCODER = tiktoken.get_encoding("cl100k_base")
    return _ENCODER


def count_tokens(text: str) -> int:
    if not text:
        return 0
    return len(_encoder().encode(text))


def fit_to_budget(
    blocks: list[ContextBlock],
    max_tokens: int,
) -> tuple[list[ContextBlock], list[DroppedBlock]]:
    """Walk blocks in input order; keep each whose token_count fits in remaining budget.
    Returns (kept, dropped). Order preservation matters — caller should pre-sort by priority."""
    used = 0
    kept: list[ContextBlock] = []
    dropped: list[DroppedBlock] = []
    for block in blocks:
        if used + block.token_count <= max_tokens:
            kept.append(block)
            used += block.token_count
        else:
            dropped.append(
                DroppedBlock(
                    chunk_id=block.chunk_id,
                    reason=f"token_budget ({used}+{block.token_count}>{max_tokens})",
                    token_count=block.token_count,
                )
            )
    return kept, dropped
