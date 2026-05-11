from __future__ import annotations

from statistics import mean

from backend.answering.models.answer import AnswerConfidence, HallucinationCheck
from backend.context.validators import GroundedContext


GROUNDING_WEIGHT = 0.4
HALLUCINATION_WEIGHT = 0.3
CITATION_WEIGHT = 0.2
RERANK_WEIGHT = 0.1


def compute_rerank_confidence(grounded_context: GroundedContext) -> float:
    scores = [block.rerank_score for block in grounded_context.context_blocks if block.rerank_score]
    if not scores:
        return 0.0
    return round(mean(scores), 4)


def compute_answer_confidence(
    grounded_context: GroundedContext,
    hallucination: HallucinationCheck,
    citation_coverage: float,
) -> AnswerConfidence:
    grounding = float(grounded_context.grounding_confidence or 0.0)
    hallucination_safety = 1.0 - float(hallucination.hallucination_risk or 0.0)
    rerank = compute_rerank_confidence(grounded_context)

    answer = (
        GROUNDING_WEIGHT * grounding
        + HALLUCINATION_WEIGHT * hallucination_safety
        + CITATION_WEIGHT * float(citation_coverage or 0.0)
        + RERANK_WEIGHT * rerank
    )

    return AnswerConfidence(
        answer_confidence=round(max(0.0, min(1.0, answer)), 4),
        grounding_confidence=round(grounding, 4),
        hallucination_risk=round(float(hallucination.hallucination_risk or 0.0), 4),
        citation_coverage=round(float(citation_coverage or 0.0), 4),
        rerank_confidence=rerank,
    )
