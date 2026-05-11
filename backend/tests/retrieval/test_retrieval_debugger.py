from backend.retrieval.retrieval_debugger import build_explanation


def test_explanation_captures_metadata_and_section_match() -> None:
    chunk = {"page_type": "Admissions", "section_type": "eligibility"}
    breakdown = {"bm25": 8.0, "bm25_normalized": 1.0, "priority": 0.95, "section_match": 1.0, "final": 0.895}
    explanation = build_explanation(
        intent="eligibility_query",
        chunk=chunk,
        score_breakdown=breakdown,
        matched_query_terms=["eligibility", "mca"],
        allowed_page_types=["Admissions", "Programs"],
        allowed_section_types=["eligibility", "admissions"],
    )
    assert explanation.intent == "eligibility_query"
    assert explanation.metadata_boost == "Admissions"
    assert explanation.section_match == "eligibility"
    assert explanation.page_type_match is True
    assert explanation.matched_terms == ["eligibility", "mca"]
    assert explanation.final_score == 0.895


def test_explanation_no_metadata_boost_when_outside_allowed() -> None:
    chunk = {"page_type": "Blog", "section_type": "general"}
    breakdown = {"bm25": 2.0, "bm25_normalized": 0.5, "priority": 0.4, "section_match": 0.0, "final": 0.43}
    explanation = build_explanation(
        intent="eligibility_query",
        chunk=chunk,
        score_breakdown=breakdown,
        matched_query_terms=[],
        allowed_page_types=["Admissions", "Programs"],
        allowed_section_types=["eligibility"],
    )
    assert explanation.metadata_boost is None
    assert explanation.section_match is None
    assert explanation.page_type_match is False
