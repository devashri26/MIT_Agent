from __future__ import annotations


# 82% of the corpus has section_type='overview' because Phase 2's section typer can't
# determine anything more specific from the chunk headings. Treating these meta-default
# types as a "topic" caps the entire corpus to N results per query, destroying retrieval.
# The cap only applies to meaningful section types.
GENERIC_SECTION_TYPES = {"overview", "general"}


def diversity_rejection_reason(
    section_type: str,
    document_id: str,
    section_counts: dict[str, int],
    document_counts: dict[str, int],
    max_per_section_type: int,
    max_per_document: int,
) -> str | None:
    """Decide if adding (section_type, document_id) would violate diversity caps."""
    if (
        section_type not in GENERIC_SECTION_TYPES
        and section_counts.get(section_type, 0) >= max_per_section_type
    ):
        return f"section_type_saturated:{section_type}"
    if document_counts.get(document_id, 0) >= max_per_document:
        return "document_saturated"
    return None
