from __future__ import annotations

from backend.context.citation_builder import build_citation
from backend.context.context_deduplicator import deduplicate_context_blocks
from backend.context.grounding import (
    DEFAULT_MIN_BLOCKS,
    DEFAULT_MIN_GROUNDING_CONFIDENCE,
    validate_grounded_context,
)
from backend.context.prompt_assembler import assemble_prompt
from backend.context.semantic_grouping import diversity_stats
from backend.context.token_budget import count_tokens, fit_to_budget
from backend.context.validators import ContextBlock, GroundedContext
from backend.reranking.validators import RerankedChunk


def _chunk_to_block(chunk: RerankedChunk) -> ContextBlock:
    return ContextBlock(
        chunk_id=chunk.chunk_id,
        document_id=chunk.document_id,
        text=chunk.text,
        citation=build_citation(chunk),
        source_url=chunk.url,
        title=chunk.title,
        page_type=chunk.page_type,
        section_type=chunk.section_type,
        section_path=list(chunk.section_path or []),
        rerank_score=chunk.rerank_score,
        answerability_score=chunk.answerability_score,
        final_relevance=chunk.final_relevance,
        token_count=count_tokens(chunk.text),
    )


def build_grounded_context(
    query: str,
    intent: str,
    reranked: list[RerankedChunk],
    token_budget: int = 2000,
    min_confidence: float = DEFAULT_MIN_GROUNDING_CONFIDENCE,
    min_blocks: int = DEFAULT_MIN_BLOCKS,
) -> GroundedContext:
    """Reranked chunks → deduplicated → token-budgeted → grounding-validated → prompt."""
    if not reranked:
        return GroundedContext(
            query=query,
            intent=intent,
            grounding_confidence=0.0,
            grounding_warnings=["no_reranked_candidates"],
            token_budget=token_budget,
        )

    blocks = [_chunk_to_block(chunk) for chunk in reranked]
    blocks, dedup_dropped = deduplicate_context_blocks(blocks)
    blocks, budget_dropped = fit_to_budget(blocks, max_tokens=token_budget)
    confidence, warnings = validate_grounded_context(
        blocks, min_confidence=min_confidence, min_blocks=min_blocks
    )
    stats = diversity_stats(blocks)

    return GroundedContext(
        query=query,
        intent=intent,
        context_blocks=blocks,
        grounding_confidence=confidence,
        grounding_warnings=warnings,
        total_tokens=sum(block.token_count for block in blocks),
        token_budget=token_budget,
        distinct_section_types=stats["distinct_section_types"],
        distinct_documents=stats["distinct_documents"],
        prompt=assemble_prompt(query, blocks),
        dropped_blocks=dedup_dropped + budget_dropped,
    )
