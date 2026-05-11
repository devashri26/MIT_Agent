from __future__ import annotations

import regex as re

from backend.answering.models.answer import AnswerCitation
from backend.context.validators import ContextBlock


_CITATION_PATTERN = re.compile(r"\[(\d+)\]")


def extract_cited_indices(answer_text: str) -> list[int]:
    """Return ordered, de-duped 1-based indices found in [N] markers."""
    seen: set[int] = set()
    order: list[int] = []
    for match in _CITATION_PATTERN.finditer(answer_text or ""):
        idx = int(match.group(1))
        if idx not in seen:
            seen.add(idx)
            order.append(idx)
    return order


def build_citations(
    answer_text: str,
    context_blocks: list[ContextBlock],
) -> list[AnswerCitation]:
    """Walk [N] markers in the answer and project them onto the corresponding ContextBlock.
    Out-of-range indices are silently skipped — the answer_validator surfaces them."""
    citations: list[AnswerCitation] = []
    for idx in extract_cited_indices(answer_text):
        if 1 <= idx <= len(context_blocks):
            block = context_blocks[idx - 1]
            citations.append(
                AnswerCitation(
                    index=idx,
                    chunk_id=block.chunk_id,
                    source_url=block.source_url,
                    title=block.title,
                    section_path=list(block.section_path or []),
                )
            )
    return citations
