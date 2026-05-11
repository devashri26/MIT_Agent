from __future__ import annotations

from backend.answering.citation_formatter import extract_cited_indices
from backend.context.validators import ContextBlock


def validate_answer(
    answer_text: str,
    context_blocks: list[ContextBlock],
) -> tuple[float, list[str]]:
    """Return (citation_coverage, warnings).

    citation_coverage = fraction of context blocks the answer cited at least once.
    Warnings flag out-of-range citation markers and missing citations entirely.
    """
    warnings: list[str] = []
    if not context_blocks:
        return 0.0, ["no_context_blocks"]

    cited = extract_cited_indices(answer_text)
    if not cited:
        warnings.append("no_citations_in_answer")

    in_range = [idx for idx in cited if 1 <= idx <= len(context_blocks)]
    out_of_range = [idx for idx in cited if idx not in in_range]
    if out_of_range:
        warnings.append(f"out_of_range_citations:{','.join(str(x) for x in out_of_range)}")

    coverage = len(set(in_range)) / max(len(context_blocks), 1)
    return round(coverage, 4), warnings
