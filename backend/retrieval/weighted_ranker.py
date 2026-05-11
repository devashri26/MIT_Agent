from typing import Any

BM25_WEIGHT = 0.7
PRIORITY_WEIGHT = 0.2
SECTION_MATCH_WEIGHT = 0.1


def normalize_scores(scores: list[float]) -> list[float]:
    """Max-normalize a list of scores to [0, 1]. Returns zeros if max is 0 or list is empty."""
    if not scores:
        return []
    max_score = max(scores)
    if max_score <= 0:
        return [0.0 for _ in scores]
    return [score / max_score for score in scores]


def section_match_bonus(section_type: str | None, allowed_section_types: list[str]) -> float:
    if not allowed_section_types or section_type is None:
        return 0.0
    return 1.0 if section_type in allowed_section_types else 0.0


def combine(
    bm25_normalized: float,
    retrieval_priority: float,
    section_match: float,
) -> float:
    return (
        BM25_WEIGHT * bm25_normalized
        + PRIORITY_WEIGHT * retrieval_priority
        + SECTION_MATCH_WEIGHT * section_match
    )


def rank_candidates(
    candidates: list[tuple[int, float]],
    chunks: list[dict[str, Any]],
    allowed_section_types: list[str],
) -> list[tuple[int, dict[str, float]]]:
    """Rank (index, bm25_score) tuples by weighted final score.

    Returns list of (chunk_index, score_breakdown) sorted by final_score descending,
    where score_breakdown has keys: bm25, bm25_normalized, priority, section_match, final.
    """
    if not candidates:
        return []
    raw_scores = [score for _, score in candidates]
    normalized = normalize_scores(raw_scores)

    ranked: list[tuple[int, dict[str, float]]] = []
    for (chunk_idx, raw), norm in zip(candidates, normalized):
        chunk = chunks[chunk_idx]
        priority = float(chunk.get("retrieval_priority", 0.0))
        section_match = section_match_bonus(chunk.get("section_type"), allowed_section_types)
        final = combine(norm, priority, section_match)
        ranked.append(
            (
                chunk_idx,
                {
                    "bm25": round(raw, 4),
                    "bm25_normalized": round(norm, 4),
                    "priority": round(priority, 4),
                    "section_match": round(section_match, 4),
                    "final": round(final, 4),
                },
            )
        )
    ranked.sort(key=lambda item: item[1]["final"], reverse=True)
    return ranked
