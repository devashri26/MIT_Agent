from __future__ import annotations

from statistics import mean
from typing import Any


def compute_grounding_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {"n": 0}
    return {
        "n": len(records),
        "abstention_correctness_rate": round(
            sum(1 for r in records if r["abstention_correct"]) / len(records), 4
        ),
        "avg_citation_correctness": round(
            mean(r["citation_correctness"] for r in records), 4
        ),
        "avg_keyword_coverage": round(mean(r["keyword_coverage"] for r in records), 4),
        "avg_grounding_confidence": round(
            mean(r["grounding_confidence"] for r in records), 4
        ),
        "avg_answer_confidence": round(mean(r["answer_confidence"] for r in records), 4),
        "fraction_abstained": round(
            sum(1 for r in records if r["abstained"]) / len(records), 4
        ),
    }
