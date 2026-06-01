from __future__ import annotations

from backend.reranking.reranker_model import DEFAULT_RERANKER_MODEL, RerankerModel
from backend.reranking.rerank_service import RerankService
from backend.reranking.validators import RerankedSearchResponse
from backend.retrieval.hybrid_retrieval import HybridRetrievalService


class RerankedRetrievalService:
    """Hybrid retrieval → cross-encoder rerank → diversity-filtered top-K.

    The reranker model is loaded lazily on the first search so importing this module is
    cheap. Inject a fake reranker in tests to skip the bge-reranker-base download.
    """

    def __init__(
        self,
        hybrid: HybridRetrievalService | None = None,
        reranker_model: object | None = None,
        candidate_pool: int = 10,
        rerank_model_name: str = DEFAULT_RERANKER_MODEL,
    ) -> None:
        self.hybrid = hybrid or HybridRetrievalService()
        self._reranker_model = reranker_model
        self._rerank_service: RerankService | None = None
        self.candidate_pool = candidate_pool
        self.rerank_model_name = rerank_model_name

    def _get_rerank_service(self) -> RerankService:
        if self._rerank_service is None:
            model = self._reranker_model or RerankerModel(model_name=self.rerank_model_name)
            self._rerank_service = RerankService(model=model)
            if hasattr(model, "model_name"):
                self.rerank_model_name = getattr(model, "model_name")
        return self._rerank_service

    def search(
        self,
        query: str,
        top_k: int = 5,
        include_components: bool = False,
        candidate_pool: int | None = None,
        max_per_section_type: int = 2,
        max_per_document: int = 2,
        duplicate_threshold: int = 85,
    ) -> RerankedSearchResponse:
        pool_size = candidate_pool or self.candidate_pool
        hybrid_response = self.hybrid.search(
            query, top_k=pool_size, include_components=include_components
        )

        kept, rejected = self._get_rerank_service().rerank(
            query=query,
            candidates=hybrid_response.results,
            top_k=top_k,
            max_per_section_type=max_per_section_type,
            max_per_document=max_per_document,
            duplicate_threshold=duplicate_threshold,
        )

        return RerankedSearchResponse(
            query=query,
            top_k=top_k,
            intent=hybrid_response.intent,
            allowed_page_types=hybrid_response.allowed_page_types,
            allowed_section_types=hybrid_response.allowed_section_types,
            expanded_terms=hybrid_response.expanded_terms,
            filter_fallback_used=hybrid_response.filter_fallback_used,
            components_excluded=hybrid_response.components_excluded,
            candidate_pool=pool_size,
            rerank_model=self.rerank_model_name,
            results=kept,
            rejected=rejected,
        )
