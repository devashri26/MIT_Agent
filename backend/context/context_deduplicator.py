from __future__ import annotations

from backend.context.validators import ContextBlock, DroppedBlock
from backend.reranking.duplicate_suppressor import suppress_semantic_duplicates


def deduplicate_context_blocks(
    blocks: list[ContextBlock],
    similarity_threshold: int = 85,
) -> tuple[list[ContextBlock], list[DroppedBlock]]:
    """Final-stage dedup. Rerank already removed near-duplicates from its candidate pool;
    this pass catches any near-duplicates that survived diversity caps or were introduced
    by a different candidate ordering."""
    if not blocks:
        return [], []
    decisions = suppress_semantic_duplicates(
        [{"chunk_id": block.chunk_id, "text": block.text} for block in blocks],
        similarity_threshold=similarity_threshold,
    )
    kept: list[ContextBlock] = []
    dropped: list[DroppedBlock] = []
    for block, duplicate_of in zip(blocks, decisions):
        if duplicate_of is None:
            kept.append(block)
        else:
            dropped.append(
                DroppedBlock(
                    chunk_id=block.chunk_id,
                    reason=f"duplicate_of:{duplicate_of}",
                    token_count=block.token_count,
                )
            )
    return kept, dropped
