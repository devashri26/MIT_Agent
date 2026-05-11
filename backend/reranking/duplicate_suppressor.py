from __future__ import annotations

from rapidfuzz import fuzz


def suppress_semantic_duplicates(
    candidates: list[dict],
    similarity_threshold: int = 85,
    text_prefix: int = 600,
) -> list[str | None]:
    """For each candidate (in order), return the chunk_id of the earlier candidate it
    duplicates, or None if unique. Uses rapidfuzz token_set_ratio on a text prefix —
    handles reordered/restated near-duplicates as well as exact repeats.
    """
    accepted: list[tuple[str, str]] = []
    decisions: list[str | None] = []
    for candidate in candidates:
        text = (candidate.get("text") or "")[:text_prefix]
        chunk_id = candidate.get("chunk_id", "")
        duplicate_of: str | None = None
        for accepted_id, accepted_text in accepted:
            ratio = fuzz.token_set_ratio(text, accepted_text)
            if ratio >= similarity_threshold:
                duplicate_of = accepted_id
                break
        if duplicate_of is None:
            accepted.append((chunk_id, text))
        decisions.append(duplicate_of)
    return decisions
