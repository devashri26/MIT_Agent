from __future__ import annotations

from typing import Any

from qdrant_client import QdrantClient

from backend.embeddings.embedding_model import EmbeddingModel
from backend.retrieval.intent_router import IntentRouter
from backend.retrieval.models.search import RetrievedChunk, SearchResponse
from backend.retrieval.query_expansion import expand_query
from backend.retrieval.retrieval_debugger import build_explanation
from backend.vectorstore.filters import build_intent_filter
from backend.vectorstore.qdrant_client import DEFAULT_COLLECTION, get_qdrant_client
from backend.vectorstore.search import qdrant_search


class DenseRetrievalService:
    """Vector retrieval via Qdrant + BAAI/bge-small-en-v1.5. Routing + payload filters are
    applied at the database level; results are post-projected into the shared RetrievedChunk
    shape so the API surface matches BM25."""

    def __init__(
        self,
        client: QdrantClient | None = None,
        model: EmbeddingModel | None = None,
        collection_name: str = DEFAULT_COLLECTION,
    ) -> None:
        self.client = client or get_qdrant_client()
        self.model = model or EmbeddingModel()
        self.collection_name = collection_name
        self.router = IntentRouter()

    def search(
        self,
        query: str,
        top_k: int = 5,
        include_components: bool = False,
        candidate_pool: int = 20,
    ) -> SearchResponse:
        route = self.router.route(query)
        _, expanded_tokens = expand_query(query)
        query_vector = self.model.embed_query(query).tolist()

        flt = build_intent_filter(
            allowed_page_types=route.allowed_page_types,
            exclude_reusable_components=not include_components,
            exclude_contaminated=True,
        )

        limit = max(top_k, candidate_pool)
        hits = qdrant_search(
            self.client,
            query_vector=query_vector,
            collection_name=self.collection_name,
            query_filter=flt,
            limit=limit,
        )

        filter_fallback_used = False
        if not hits and flt is not None:
            relaxed = build_intent_filter(
                allowed_page_types=None,
                exclude_reusable_components=not include_components,
                exclude_contaminated=True,
            )
            hits = qdrant_search(
                self.client,
                query_vector=query_vector,
                collection_name=self.collection_name,
                query_filter=relaxed,
                limit=limit,
            )
            filter_fallback_used = True

        results = [
            self._hit_to_result(hit, rank=rank, route=route)
            for rank, hit in enumerate(hits[:top_k], start=1)
        ]

        return SearchResponse(
            query=query,
            top_k=top_k,
            intent=route.intent,
            allowed_page_types=route.allowed_page_types,
            allowed_section_types=route.allowed_section_types,
            expanded_terms=expanded_tokens,
            filter_fallback_used=filter_fallback_used,
            components_excluded=0,
            results=results,
        )

    def search_candidate_chunk_ids(
        self,
        query: str,
        candidate_pool: int = 20,
        include_components: bool = False,
    ) -> tuple[list[str], dict[str, float]]:
        """Return ordered chunk_ids + score lookup for fusion. Does not project results."""
        route = self.router.route(query)
        query_vector = self.model.embed_query(query).tolist()
        flt = build_intent_filter(
            allowed_page_types=route.allowed_page_types,
            exclude_reusable_components=not include_components,
            exclude_contaminated=True,
        )
        hits = qdrant_search(
            self.client,
            query_vector=query_vector,
            collection_name=self.collection_name,
            query_filter=flt,
            limit=candidate_pool,
        )
        if not hits and flt is not None:
            relaxed = build_intent_filter(
                allowed_page_types=None,
                exclude_reusable_components=not include_components,
                exclude_contaminated=True,
            )
            hits = qdrant_search(
                self.client,
                query_vector=query_vector,
                collection_name=self.collection_name,
                query_filter=relaxed,
                limit=candidate_pool,
            )
        chunk_ids = [hit.payload["chunk_id"] for hit in hits]
        scores = {hit.payload["chunk_id"]: float(hit.score) for hit in hits}
        return chunk_ids, scores

    def _hit_to_result(self, hit: Any, rank: int, route: Any) -> RetrievedChunk:
        payload = hit.payload or {}
        score = float(hit.score)
        breakdown = {
            "bm25": 0.0,
            "bm25_normalized": 0.0,
            "priority": float(payload.get("retrieval_priority") or 0.0),
            "section_match": (
                1.0 if payload.get("section_type") in (route.allowed_section_types or []) else 0.0
            ),
            "final": round(score, 4),
        }
        explanation = build_explanation(
            intent=route.intent,
            chunk=payload,
            score_breakdown=breakdown,
            matched_query_terms=[],
            allowed_page_types=route.allowed_page_types,
            allowed_section_types=route.allowed_section_types,
        )
        return RetrievedChunk(
            rank=rank,
            score=round(score, 4),
            chunk_id=payload["chunk_id"],
            document_id=payload["document_id"],
            url=payload["url"],
            canonical_url=payload.get("canonical_url") or payload["url"],
            title=payload["title"],
            department=payload.get("department"),
            page_type=payload["page_type"],
            section_type=payload.get("section_type") or "general",
            section_heading=payload.get("section_heading") or "",
            section_path=payload.get("section_path") or [],
            token_count=int(payload.get("token_count") or 0),
            content_type=payload.get("content_type") or "",
            quality_score=float(payload.get("quality_score") or 0.0),
            retrieval_priority=float(payload.get("retrieval_priority") or 0.0),
            quality_flags=payload.get("quality_flags") or [],
            is_reusable_component=bool(payload.get("is_reusable_component")),
            component_type=payload.get("component_type"),
            mixed_topic=bool(payload.get("mixed_topic")),
            dominant_topics=payload.get("dominant_topics") or [],
            cross_domain_contamination=bool(payload.get("cross_domain_contamination")),
            contamination_sources=payload.get("contamination_sources") or [],
            text=payload.get("text") or "",
            explanation=explanation,
            metadata={"dense_similarity": round(score, 4)},
        )
