from collections import Counter

from backend.context.validators import ContextBlock


def diversity_stats(blocks: list[ContextBlock]) -> dict:
    """Compute topic diversity stats over a list of context blocks."""
    if not blocks:
        return {
            "distinct_section_types": 0,
            "distinct_documents": 0,
            "section_type_distribution": {},
            "page_type_distribution": {},
        }
    return {
        "distinct_section_types": len({b.section_type for b in blocks}),
        "distinct_documents": len({b.document_id for b in blocks}),
        "section_type_distribution": dict(Counter(b.section_type for b in blocks)),
        "page_type_distribution": dict(Counter(b.page_type for b in blocks)),
    }
