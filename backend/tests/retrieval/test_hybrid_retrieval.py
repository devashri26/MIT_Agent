from pathlib import Path

import orjson
import pytest

from backend.retrieval.bm25_service import BM25RetrievalService
from backend.retrieval.dense_retrieval import DenseRetrievalService
from backend.retrieval.hybrid_retrieval import HybridRetrievalService
from backend.tests.conftest import FakeEmbeddingModel
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
        "section_path": [page_type],
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
def hybrid_service(tmp_path: Path, fake_model: FakeEmbeddingModel) -> HybridRetrievalService:
    records = [
        _record("c1", "Admissions", "eligibility", "MCA eligibility minimum qualification entrance exam", fake_model),
        _record("c2", "Programs", "curriculum", "BTech curriculum semester credits electives core", fake_model),
        _record("c3", "Blog", "general", "Generic blog post about exam tips and student life", fake_model),
        _record("c4", "Admissions", "admissions", "MCA admissions apply process steps and timeline", fake_model),
    ]
    embedded_path = tmp_path / "embedded.jsonl"
    embedded_path.write_bytes(b"\n".join(orjson.dumps(r) for r in records))

    chunks_path = tmp_path / "chunks.jsonl"
    chunks_path.write_bytes(
        b"\n".join(orjson.dumps({k: v for k, v in r.items() if k != "embedding"}) for r in records)
    )

    qdrant = get_qdrant_client(in_memory=True)
    ingest_embeddings_to_qdrant(qdrant, embedded_path, collection_name="hybrid_test")

    bm25 = BM25RetrievalService(chunks_path=chunks_path)
    dense = DenseRetrievalService(client=qdrant, model=fake_model, collection_name="hybrid_test")
    return HybridRetrievalService(bm25=bm25, dense=dense, candidate_pool=10)


def test_hybrid_search_returns_results(hybrid_service) -> None:
    response = hybrid_service.search("MCA eligibility minimum qualification entrance exam", top_k=3)
    assert response.intent == "eligibility_query"
    assert len(response.results) >= 1


def test_hybrid_result_carries_source_signals(hybrid_service) -> None:
    response = hybrid_service.search("MCA eligibility minimum qualification entrance exam", top_k=3)
    top = response.results[0]
    assert top.fusion_score > 0
    assert top.retrieval_source, "expected at least one retrieval source"
    # When BM25 surfaced it, bm25_rank > 0; when dense surfaced it, dense_rank > 0
    assert top.bm25_rank > 0 or top.dense_rank > 0


def test_hybrid_excludes_components_by_default(tmp_path: Path, fake_model: FakeEmbeddingModel) -> None:
    records = [
        _record("c1", "Admissions", "eligibility", "MCA eligibility text", fake_model, is_reusable_component=True),
        _record("c2", "Admissions", "eligibility", "MCA eligibility text alternate version", fake_model),
    ]
    embedded_path = tmp_path / "embedded.jsonl"
    chunks_path = tmp_path / "chunks.jsonl"
    embedded_path.write_bytes(b"\n".join(orjson.dumps(r) for r in records))
    chunks_path.write_bytes(b"\n".join(orjson.dumps({k: v for k, v in r.items() if k != "embedding"}) for r in records))

    qdrant = get_qdrant_client(in_memory=True)
    ingest_embeddings_to_qdrant(qdrant, embedded_path, collection_name="t")

    bm25 = BM25RetrievalService(chunks_path=chunks_path)
    dense = DenseRetrievalService(client=qdrant, model=fake_model, collection_name="t")
    hybrid = HybridRetrievalService(bm25=bm25, dense=dense, candidate_pool=5)

    response = hybrid.search("MCA eligibility", top_k=5)
    assert all(r.chunk_id != "c1" for r in response.results)
