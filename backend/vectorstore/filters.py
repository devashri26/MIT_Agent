from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue


def build_intent_filter(
    allowed_page_types: list[str] | None = None,
    allowed_section_types: list[str] | None = None,
    exclude_reusable_components: bool = True,
    exclude_contaminated: bool = True,
    min_quality_score: float | None = None,
) -> Filter | None:
    """Build a Qdrant Filter from intent routing + safety flags. Returns None when no
    constraints are active so callers can pass the result directly."""
    must: list = []
    must_not: list = []

    if allowed_page_types:
        must.append(FieldCondition(key="page_type", match=MatchAny(any=list(allowed_page_types))))
    if allowed_section_types:
        must.append(
            FieldCondition(key="section_type", match=MatchAny(any=list(allowed_section_types)))
        )
    if exclude_reusable_components:
        must_not.append(
            FieldCondition(key="is_reusable_component", match=MatchValue(value=True))
        )
    if exclude_contaminated:
        must_not.append(
            FieldCondition(key="cross_domain_contamination", match=MatchValue(value=True))
        )

    if not must and not must_not:
        return None
    return Filter(must=must or None, must_not=must_not or None)
