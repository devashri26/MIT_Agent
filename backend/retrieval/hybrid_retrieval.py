from __future__ import annotations

from backend.retrieval.bm25_service import BM25RetrievalService
from backend.retrieval.dense_retrieval import DenseRetrievalService
from backend.retrieval.fusion import DEFAULT_RRF_K, reciprocal_rank_fusion
from backend.retrieval.models.search import RetrievedChunk, SearchResponse


class HybridRetrievalService:
    """BM25 + dense retrieval combined via reciprocal rank fusion.

    Each retriever runs independently with its existing routing + filtering. Their top-N
    candidate lists are fused with RRF; the merged top-K is projected back into
    RetrievedChunk records. BM25's RetrievedChunk wins on conflict because it carries the
    matched_terms explanation; dense-only chunks fall back to the dense projection.
    """

    def __init__(
        self,
        bm25: BM25RetrievalService | None = None,
        dense: DenseRetrievalService | None = None,
        candidate_pool: int = 20,
        rrf_k: int = DEFAULT_RRF_K,
    ) -> None:
        self.bm25 = bm25 or BM25RetrievalService()
        self.dense = dense or DenseRetrievalService()
        self.candidate_pool = candidate_pool
        self.rrf_k = rrf_k

    def search(
        self,
        query: str,
        top_k: int = 5,
        include_components: bool = False,
    ) -> SearchResponse:
        bm25_response = self.bm25.search(
            query, top_k=self.candidate_pool, include_components=include_components
        )
        dense_response = self.dense.search(
            query, top_k=self.candidate_pool, include_components=include_components
        )

        chunk_map: dict[str, RetrievedChunk] = {}
        for result in dense_response.results:
            chunk_map[result.chunk_id] = result
        for result in bm25_response.results:
            chunk_map[result.chunk_id] = result

        bm25_ids = [r.chunk_id for r in bm25_response.results]
        dense_ids = [r.chunk_id for r in dense_response.results]
        fused = reciprocal_rank_fusion(bm25_ids, dense_ids, k=self.rrf_k)

        results: list[RetrievedChunk] = []
        for rank, item in enumerate(fused[:top_k], start=1):
            base = chunk_map.get(item.chunk_id)
            if base is None:
                continue
            results.append(
                base.model_copy(
                    update={
                        "rank": rank,
                        "score": round(item.fusion_score, 6),
                        "retrieval_source": item.sources,
                        "bm25_rank": item.bm25_rank,
                        "dense_rank": item.dense_rank,
                        "fusion_score": round(item.fusion_score, 6),
                    }
                )
            )

        return SearchResponse(
            query=query,
            top_k=top_k,
            intent=bm25_response.intent,
            allowed_page_types=bm25_response.allowed_page_types,
            allowed_section_types=bm25_response.allowed_section_types,
            expanded_terms=bm25_response.expanded_terms,
            filter_fallback_used=bm25_response.filter_fallback_used
            or dense_response.filter_fallback_used,
            components_excluded=bm25_response.components_excluded,
            results=results,
        )
