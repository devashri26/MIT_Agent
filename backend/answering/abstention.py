from __future__ import annotations

from backend.answering.models.answer import HallucinationCheck
from backend.context.validators import GroundedContext


def should_abstain(
    grounded_context: GroundedContext,
    hallucination: HallucinationCheck | None = None,
    min_grounding_confidence: float = 0.3,
    max_hallucination_risk: float = 0.5,
) -> tuple[bool, str | None]:
    """Decide whether to abstain before exposing an answer.

    Order matters: hard signals first (no blocks → no answer), then confidence, then
    post-generation hallucination risk. Returns (abstain, reason)."""
    if not grounded_context.context_blocks:
        return True, "no_context_blocks"
    if "no_blocks" in grounded_context.grounding_warnings:
        return True, "no_blocks_warning"
    if grounded_context.grounding_confidence < min_grounding_confidence:
        return True, f"low_grounding_confidence:{grounded_context.grounding_confidence:.2f}"
    if hallucination is not None and hallucination.hallucination_risk >= max_hallucination_risk:
        return True, f"high_hallucination_risk:{hallucination.hallucination_risk:.2f}"
    return False, None
