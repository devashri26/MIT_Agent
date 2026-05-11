from __future__ import annotations

import math


def sigmoid(value: float) -> float:
    """Numerically stable sigmoid."""
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def calibrate(raw_score: float) -> float:
    """Map cross-encoder logit to [0, 1]. bge-reranker-base outputs are roughly in [-10, 10];
    sigmoid puts the meaningful range in [0.05, 0.95]."""
    return round(sigmoid(float(raw_score)), 4)


def combine_relevance(
    rerank_calibrated: float,
    answerability: float,
    rerank_weight: float = 0.8,
    answerability_weight: float = 0.2,
) -> float:
    """Default blend for ranking. Returns clamped [0,1]."""
    total = rerank_weight + answerability_weight
    if total <= 0:
        return 0.0
    score = (rerank_weight * rerank_calibrated + answerability_weight * answerability) / total
    return round(max(0.0, min(1.0, score)), 4)
