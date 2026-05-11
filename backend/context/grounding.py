from __future__ import annotations

from backend.context.validators import ContextBlock


DEFAULT_MIN_GROUNDING_CONFIDENCE = 0.5
DEFAULT_MIN_BLOCKS = 2


def compute_grounding_confidence(blocks: list[ContextBlock]) -> float:
    if not blocks:
        return 0.0
    total = sum(block.final_relevance for block in blocks)
    return round(total / len(blocks), 4)


def validate_grounded_context(
    blocks: list[ContextBlock],
    min_confidence: float = DEFAULT_MIN_GROUNDING_CONFIDENCE,
    min_blocks: int = DEFAULT_MIN_BLOCKS,
) -> tuple[float, list[str]]:
    """Compute grounding_confidence and surface warnings the caller should consider before
    handing context to an LLM. Does not reject anything — that's the caller's policy."""
    warnings: list[str] = []
    if not blocks:
        return 0.0, ["no_blocks"]

    confidence = compute_grounding_confidence(blocks)

    if confidence < min_confidence:
        warnings.append(f"low_confidence:{confidence:.2f}<{min_confidence:.2f}")
    if len(blocks) < min_blocks:
        warnings.append(f"insufficient_blocks:{len(blocks)}<{min_blocks}")

    distinct_section_types = len({block.section_type for block in blocks})
    if len(blocks) >= 3 and distinct_section_types < 2:
        warnings.append(f"low_topical_diversity:{distinct_section_types}")

    return confidence, warnings
