from pydantic import BaseModel, Field


class RetrievalExplanation(BaseModel):
    intent: str
    matched_terms: list[str] = Field(default_factory=list)
    metadata_boost: str | None = None
    section_match: str | None = None
    retrieval_priority: float = 0.0
    bm25_score: float = 0.0
    bm25_normalized: float = 0.0
    section_match_bonus: float = 0.0
    page_type_match: bool = False
    final_score: float = 0.0


def build_explanation(
    intent: str,
    chunk: dict,
    score_breakdown: dict[str, float],
    matched_query_terms: list[str],
    allowed_page_types: list[str],
    allowed_section_types: list[str],
) -> RetrievalExplanation:
    page_type = chunk.get("page_type")
    section_type = chunk.get("section_type")
    return RetrievalExplanation(
        intent=intent,
        matched_terms=matched_query_terms,
        metadata_boost=page_type if page_type in allowed_page_types else None,
        section_match=section_type if section_type in allowed_section_types else None,
        retrieval_priority=score_breakdown.get("priority", 0.0),
        bm25_score=score_breakdown.get("bm25", 0.0),
        bm25_normalized=score_breakdown.get("bm25_normalized", 0.0),
        section_match_bonus=score_breakdown.get("section_match", 0.0),
        page_type_match=page_type in allowed_page_types,
        final_score=score_breakdown.get("final", 0.0),
    )
