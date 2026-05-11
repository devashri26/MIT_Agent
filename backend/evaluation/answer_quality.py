from __future__ import annotations

from typing import Any

from backend.answering.models.answer import GroundedAnswer


def evaluate_answer_quality(answer: GroundedAnswer, expected: dict[str, Any]) -> dict[str, Any]:
    """Compare a generated answer against an expected QA entry. Substring-match expected
    URL fragments rather than exact equality so partial URL knowledge is enough."""
    expected_abstain = bool(expected.get("expected_abstention", False))
    abstention_correct = answer.abstained == expected_abstain

    expected_substrings = [s.lower() for s in expected.get("expected_source_url_substrings") or [] if s]
    actual_urls = [c.source_url.lower() for c in answer.citations]
    matched_substrings = sum(
        1 for sub in expected_substrings if any(sub in url for url in actual_urls)
    )
    citation_correctness = (
        matched_substrings / len(expected_substrings) if expected_substrings else 1.0
    )

    expected_keywords = [k.lower() for k in expected.get("expected_keywords") or [] if k]
    answer_text_lower = (answer.answer or "").lower()
    keyword_hits = sum(1 for k in expected_keywords if k in answer_text_lower)
    keyword_coverage = (
        keyword_hits / len(expected_keywords) if expected_keywords else 1.0
    )

    return {
        "id": expected.get("id"),
        "query": answer.query,
        "abstained": answer.abstained,
        "expected_abstention": expected_abstain,
        "abstention_correct": abstention_correct,
        "citation_correctness": round(citation_correctness, 4),
        "keyword_coverage": round(keyword_coverage, 4),
        "expected_substrings_matched": matched_substrings,
        "expected_substrings_total": len(expected_substrings),
        "n_citations": len(answer.citations),
        "grounding_confidence": answer.confidence.grounding_confidence,
        "answer_confidence": answer.confidence.answer_confidence,
        "hallucination_risk": answer.hallucination.hallucination_risk,
        "hallucination_flagged": not answer.hallucination.safe_to_return,
        "provider": answer.provider,
        "model": answer.model,
        "abstention_reason": answer.abstention_reason,
    }
