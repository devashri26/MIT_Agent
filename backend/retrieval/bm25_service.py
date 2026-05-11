from pathlib import Path
from typing import Any

import orjson
import regex as re
from rank_bm25 import BM25Okapi

from backend.retrieval.intent_router import IntentRouter
from backend.retrieval.metadata_filters import exclude_reusable_components, filter_by_page_types
from backend.retrieval.models.search import RetrievedChunk, SearchResponse
from backend.retrieval.query_expansion import expand_query
from backend.retrieval.retrieval_debugger import build_explanation
from backend.retrieval.weighted_ranker import rank_candidates


class BM25RetrievalService:
    """Routed, weighted BM25 retrieval over normalized chunks.

    Pipeline: intent route → query expansion → metadata filter (by page_type) → BM25
    scoring against global IDF → weighted ranking (0.7 BM25 + 0.2 priority + 0.1 section)
    → explanations. Falls back to unfiltered candidates if the filter zeroes the result.
    """

    def __init__(
        self,
        chunks_path: Path = Path("datasets/normalized_chunks.jsonl"),
    ) -> None:
        self.chunks_path = chunks_path
        self.chunks = self._load_chunks(chunks_path)
        self.tokenized_corpus = [self._tokenize(self._search_text(c)) for c in self.chunks]
        self.index = BM25Okapi(self.tokenized_corpus) if self.tokenized_corpus else None
        self.router = IntentRouter()

    def search(self, query: str, top_k: int = 5, include_components: bool = False) -> SearchResponse:
        route = self.router.route(query)
        original_tokens, expanded_tokens = expand_query(query)

        if not expanded_tokens or self.index is None:
            return SearchResponse(
                query=query,
                top_k=top_k,
                intent=route.intent,
                allowed_page_types=route.allowed_page_types,
                allowed_section_types=route.allowed_section_types,
                expanded_terms=expanded_tokens,
                filter_fallback_used=False,
                components_excluded=0,
                results=[],
            )

        bm25_scores = self.index.get_scores(expanded_tokens)

        filtered_indices = filter_by_page_types(self.chunks, route.allowed_page_types)
        filter_fallback_used = False
        if not filtered_indices:
            filtered_indices = list(range(len(self.chunks)))
            filter_fallback_used = True

        components_excluded = 0
        if not include_components:
            filtered_indices, components_excluded = exclude_reusable_components(
                self.chunks, filtered_indices
            )
            if not filtered_indices:
                filtered_indices = list(range(len(self.chunks)))
                filter_fallback_used = True

        candidates = [(idx, float(bm25_scores[idx])) for idx in filtered_indices]

        ranked = rank_candidates(candidates, self.chunks, route.allowed_section_types)[:top_k]

        results: list[RetrievedChunk] = []
        for rank, (chunk_idx, breakdown) in enumerate(ranked, start=1):
            chunk = self.chunks[chunk_idx]
            chunk_tokens = set(self.tokenized_corpus[chunk_idx])
            matched_query_terms = [t for t in original_tokens if t in chunk_tokens]
            explanation = build_explanation(
                intent=route.intent,
                chunk=chunk,
                score_breakdown=breakdown,
                matched_query_terms=matched_query_terms,
                allowed_page_types=route.allowed_page_types,
                allowed_section_types=route.allowed_section_types,
            )
            results.append(
                RetrievedChunk(
                    rank=rank,
                    score=breakdown["final"],
                    chunk_id=chunk["chunk_id"],
                    document_id=chunk["document_id"],
                    url=chunk["url"],
                    canonical_url=chunk.get("canonical_url", chunk["url"]),
                    title=chunk["title"],
                    department=chunk.get("department"),
                    page_type=chunk["page_type"],
                    section_type=chunk.get("section_type", "general"),
                    section_heading=chunk["section_heading"],
                    section_path=chunk.get("section_path") or [],
                    token_count=chunk["token_count"],
                    content_type=chunk["content_type"],
                    quality_score=chunk["quality_score"],
                    retrieval_priority=float(chunk.get("retrieval_priority", 0.0)),
                    quality_flags=chunk.get("quality_flags") or [],
                    is_reusable_component=bool(chunk.get("is_reusable_component")),
                    component_type=chunk.get("component_type"),
                    mixed_topic=bool(chunk.get("mixed_topic")),
                    dominant_topics=chunk.get("dominant_topics") or [],
                    cross_domain_contamination=bool(chunk.get("cross_domain_contamination")),
                    contamination_sources=chunk.get("contamination_sources") or [],
                    text=chunk["text"],
                    explanation=explanation,
                    metadata=chunk.get("metadata") or {},
                )
            )

        return SearchResponse(
            query=query,
            top_k=top_k,
            intent=route.intent,
            allowed_page_types=route.allowed_page_types,
            allowed_section_types=route.allowed_section_types,
            expanded_terms=expanded_tokens,
            filter_fallback_used=filter_fallback_used,
            components_excluded=components_excluded,
            results=results,
        )

    @staticmethod
    def _load_chunks(path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        return [orjson.loads(line) for line in path.read_bytes().splitlines() if line.strip()]

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"[\p{L}\p{N}][\p{L}\p{N}&.+-]*", text.lower())

    @staticmethod
    def _search_text(chunk: dict[str, Any]) -> str:
        metadata = chunk.get("metadata") or {}
        headings = " ".join(metadata.get("headings") or [])
        return " ".join(
            [
                chunk.get("title") or "",
                chunk.get("section_heading") or "",
                headings,
                chunk.get("page_type") or "",
                chunk.get("section_type") or "",
                chunk.get("content_type") or "",
                chunk.get("text") or "",
            ]
        )
