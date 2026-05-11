from __future__ import annotations

from dataclasses import dataclass


DEFAULT_RRF_K = 60


@dataclass
class FusedItem:
    chunk_id: str
    fusion_score: float
    bm25_rank: int
    dense_rank: int
    sources: list[str]


def reciprocal_rank_fusion(
    bm25_results: list[str],
    dense_results: list[str],
    k: int = DEFAULT_RRF_K,
) -> list[FusedItem]:
    """Standard RRF: fusion_score = sum(1 / (k + rank)) over each retriever that surfaced
    the doc. Inputs are ordered lists of chunk_ids (rank 1 first). Output sorted by
    fusion_score descending; each chunk_id appears exactly once. rank=0 in the output means
    the doc was absent from that retriever's results."""
    bm25_ranks = {chunk_id: rank for rank, chunk_id in enumerate(bm25_results, start=1)}
    dense_ranks = {chunk_id: rank for rank, chunk_id in enumerate(dense_results, start=1)}

    scores: dict[str, float] = {}
    sources: dict[str, list[str]] = {}

    for rank, chunk_id in enumerate(bm25_results, start=1):
        scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
        sources.setdefault(chunk_id, []).append("bm25")

    for rank, chunk_id in enumerate(dense_results, start=1):
        scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
        sources.setdefault(chunk_id, []).append("dense")

    fused = [
        FusedItem(
            chunk_id=chunk_id,
            fusion_score=round(scores[chunk_id], 6),
            bm25_rank=bm25_ranks.get(chunk_id, 0),
            dense_rank=dense_ranks.get(chunk_id, 0),
            sources=sources[chunk_id],
        )
        for chunk_id in scores
    ]
    fused.sort(key=lambda f: f.fusion_score, reverse=True)
    return fused
