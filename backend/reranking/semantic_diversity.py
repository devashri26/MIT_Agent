from __future__ import annotations


def diversity_rejection_reason(
    section_type: str,
    document_id: str,
    section_counts: dict[str, int],
    document_counts: dict[str, int],
    max_per_section_type: int,
    max_per_document: int,
) -> str | None:
    """Pure helper: given current counts and per-key caps, decide if adding (section_type,
    document_id) would violate diversity caps. Returns the reason string or None."""
    if section_counts.get(section_type, 0) >= max_per_section_type:
        return f"section_type_saturated:{section_type}"
    if document_counts.get(document_id, 0) >= max_per_document:
        return "document_saturated"
    return None
