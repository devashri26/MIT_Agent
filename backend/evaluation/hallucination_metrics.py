from __future__ import annotations

from statistics import mean
from typing import Any


def compute_hallucination_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {"n": 0}
    answered = [r for r in records if not r["abstained"]]
    if not answered:
        return {
            "n": 0,
            "avg_hallucination_risk": 0.0,
            "fraction_flagged": 0.0,
            "answered_queries": 0,
        }
    return {
        "n": len(answered),
        "answered_queries": len(answered),
        "avg_hallucination_risk": round(
            mean(r["hallucination_risk"] for r in answered), 4
        ),
        "fraction_flagged": round(
            sum(1 for r in answered if r["hallucination_flagged"]) / len(answered), 4
        ),
    }
