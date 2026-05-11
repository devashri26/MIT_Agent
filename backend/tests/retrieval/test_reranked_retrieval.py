from pathlib import Path

import orjson
import pytest

from backend.retrieval.bm25_service import BM25RetrievalService
from backend.retrieval.dense_retrieval import DenseRetrievalService
from backend.retrieval.hybrid_retrieval import HybridRetrievalService
from backend.retrieval.reranked_retrieval import RerankedRetrievalService
from backend.tests.conftest import FakeEmbeddingModel, FakeRerankerModel
from backend.vectorstore.ingestion import ingest_embeddings_to_qdrant
from backend.vectorstore.qdrant_client import get_qdrant_client


def _record(chunk_id: str, page_type: str, section_type: str, text: str, model: FakeEmbeddingModel, **extra):
    vec = model.embed([text])[0]
    return {
        "chunk_id": chunk_id,
        "document_id": f"doc_{chunk_id}",
        "url": f"https://example.edu/{chunk_id}",
        "canonical_url": f"https://example.edu/{chunk_id}",
        "title": text[:30],
        "department": None,
        "section_heading": "Overview",
        "section_path": [page_type, section_type],
        "page_type": page_type,
        "page_type_confidence": 0.95,
        "section_type": section_type,
        "retrieval_priority": 0.9,
        "quality_flags": [],
        "quality_score": 1.0,
        "is_reusable_component": False,
        "component_type": None,
        "mixed_topic": False,
        "dominant_topics": [],
        "cross_domain_contamination": False,
        "contamination_sources": [],
        "token_count": len(text.split()),
        "content_type": "GENERAL",
        "embedding_model": model.model_name,
        "embedded_at": "2026-05-11T00:00:00Z",
        "text": text,
        "embedding": vec.tolist(),
        "chunk_hash": chunk_id,
        "is_canonical": True,
        "chunk_index": 0,
        **extra,
    }


@pytest.fixture
def reranked_service(tmp_path: Path, fake_model: FakeEmbeddingModel, fake_reranker: FakeRerankerModel) -> RerankedRetrievalService:
    records = [
        _record("c1", "Admissions", "eligibility", "MCA eligibility minimum qualification entrance exam degree", fake_model),
        _record("c2", "Admissions", "fees", "MCA fee structure tuition charges per semester development", fake_model),
        _record("c3", "Programs", "curriculum", "BTech curriculum semester credits electives core courses", fake_model),
        _record("c4", "Admissions", "admissions", "MCA admissions process apply timeline deadlines", fake_model),
        _record("c5", "Blog", "general", "Generic blog post about student life and campus", fake_model),
    ]
    embedded_path = tmp_path / "embedded.jsonl"
    chunks_path = tmp_path / "chunks.jsonl"
    embedded_path.write_bytes(b"\n".join(orjson.dumps(r) for r in records))
    chunks_path.write_bytes(
        b"\n".join(orjson.dumps({k: v for k, v in r.items() if k != "embedding"}) for r in records)
    )

    qdrant = get_qdrant_client(in_memory=True)
    ingest_embeddings_to_qdrant(qdrant, embedded_path, collection_name="rerank_test")

    bm25 = BM25RetrievalService(chunks_path=chunks_path)
    dense = DenseRetrievalService(client=qdrant, model=fake_model, collection_name="rerank_test")
    hybrid = HybridRetrievalService(bm25=bm25, dense=dense, candidate_pool=5)
    return RerankedRetrievalService(hybrid=hybrid, reranker_model=fake_reranker, candidate_pool=5)


def test_reranked_search_returns_top_k(reranked_service) -> None:
    response = reranked_service.search("MCA eligibility minimum qualification entrance exam", top_k=3)
    assert len(response.results) <= 3
    assert response.results[0].rerank_score > 0
    assert response.results[0].final_relevance > 0
    assert response.rerank_model == "fake/test-reranker"


def test_reranked_search_records_rejected(reranked_service) -> None:
    response = reranked_service.search("MCA eligibility minimum qualification", top_k=2, max_per_section_type=2)
    total_returned = len(response.results) + len(response.rejected)
    assert total_returned >= 1
    for rej in response.rejected:
        assert rej.rejection_reason is not None


def test_reranked_search_preserves_intent_routing(reranked_service) -> None:
    response = reranked_service.search("MCA eligibility minimum qualification", top_k=3)
    assert response.intent == "eligibility_query"
    assert "Admissions" in response.allowed_page_types
