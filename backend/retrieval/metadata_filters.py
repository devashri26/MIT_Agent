from typing import Any


def filter_by_page_types(
    chunks: list[dict[str, Any]],
    allowed_page_types: list[str] | None,
) -> list[int]:
    """Return indices of chunks whose page_type is in allowed_page_types.

    Empty/None allowed list returns all indices. The caller is responsible for
    falling back to the unfiltered corpus if the result is empty.
    """
    if not allowed_page_types:
        return list(range(len(chunks)))
    allowed = set(allowed_page_types)
    return [idx for idx, chunk in enumerate(chunks) if chunk.get("page_type") in allowed]


def exclude_reusable_components(
    chunks: list[dict[str, Any]],
    indices: list[int],
) -> tuple[list[int], int]:
    """Remove indices pointing to chunks flagged is_reusable_component=True.

    Returns (kept_indices, excluded_count).
    """
    kept: list[int] = []
    excluded = 0
    for idx in indices:
        if chunks[idx].get("is_reusable_component"):
            excluded += 1
            continue
        kept.append(idx)
    return kept, excluded
