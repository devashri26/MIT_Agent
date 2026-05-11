from pathlib import Path

import numpy as np
import orjson
import pytest

from backend.retrieval.dense_retrieval import DenseRetrievalService
from backend.tests.conftest import FakeEmbeddingModel
from backend.vectorstore.ingestion import ingest_embeddings_to_qdrant
from backend.vectorstore.qdrant_client import get_qdrant_client


def _embedded(chunk_id: str, page_type: str, text: str, model: FakeEmbeddingModel, **extra):
    vec = model.embed([text])[0]
    return {
        "chunk_id": chunk_id,
        "document_id": f"doc_{chunk_id}",
        "url": f"https://example.edu/{chunk_id}",
        "canonical_url": f"https://example.edu/{chunk_id}",
        "title": text[:40],
        "department": None,
        "section_heading": "Overview",
        "section_path": [page_type],
        "page_type": page_type,
        "page_type_confidence": 0.95,
        "section_type": "overview",
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
        **extra,
    }


@pytest.fixture
def populated_qdrant(tmp_path: Path, fake_model: FakeEmbeddingModel):
    embedded_path = tmp_path / "embedded.jsonl"
    records = [
        _embedded("c1", "Admissions", "MCA eligibility minimum qualification entrance", fake_model),
        _embedded("c2", "Programs", "BTech curriculum semester credits electives", fake_model),
        _embedded("c3", "Blog", "MCA blog post about exam tips and tricks", fake_model),
    ]
    embedded_path.write_bytes(b"\n".join(orjson.dumps(r) for r in records))
    client = get_qdrant_client(in_memory=True)
    ingest_embeddings_to_qdrant(client, embedded_path, collection_name="dense_test")
    return client


def test_dense_retrieval_returns_routed_results(populated_qdrant, fake_model) -> None:
    service = DenseRetrievalService(
        client=populated_qdrant,
        model=fake_model,
        collection_name="dense_test",
    )
    response = service.search("MCA eligibility minimum qualification entrance", top_k=3)
    assert response.intent == "eligibility_query"
    assert response.results, "expected at least one result"
    assert response.results[0].page_type in {"Admissions", "Programs"}


def test_dense_retrieval_excludes_components_by_default(tmp_path: Path, fake_model) -> None:
    embedded_path = tmp_path / "embedded.jsonl"
    records = [
        _embedded("c1", "Admissions", "MCA eligibility text", fake_model, is_reusable_component=True),
        _embedded("c2", "Admissions", "MCA eligibility text alt", fake_model),
    ]
    embedded_path.write_bytes(b"\n".join(orjson.dumps(r) for r in records))
    client = get_qdrant_client(in_memory=True)
    ingest_embeddings_to_qdrant(client, embedded_path, collection_name="t")

    service = DenseRetrievalService(client=client, model=fake_model, collection_name="t")
    response = service.search("MCA eligibility", top_k=5)
    assert all(r.chunk_id != "c1" for r in response.results)
