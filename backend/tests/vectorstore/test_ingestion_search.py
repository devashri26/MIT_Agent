from pathlib import Path

import numpy as np
import orjson

from backend.vectorstore.collection_manager import collection_count
from backend.vectorstore.filters import build_intent_filter
from backend.vectorstore.ingestion import ingest_embeddings_to_qdrant
from backend.vectorstore.qdrant_client import get_qdrant_client
from backend.vectorstore.search import qdrant_search


def _embedded_record(chunk_id: str, vector: list[float], page_type: str, **payload) -> dict:
    return {
        "chunk_id": chunk_id,
        "document_id": f"doc_{chunk_id}",
        "url": "https://example.edu/x",
        "canonical_url": "https://example.edu/x",
        "title": "Example",
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
        "token_count": 100,
        "content_type": "GENERAL",
        "embedding_model": "fake",
        "embedded_at": "2026-05-11T00:00:00Z",
        "text": "sample text",
        "embedding": vector,
        **payload,
    }


def _unit(vec: list[float]) -> list[float]:
    arr = np.array(vec, dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    if norm == 0:
        return arr.tolist()
    return (arr / norm).tolist()


def test_ingestion_creates_points(tmp_path: Path) -> None:
    embedded_path = tmp_path / "embedded.jsonl"
    records = [
        _embedded_record("c1", _unit([1.0, 0.0, 0.0, 0.0]), "Admissions"),
        _embedded_record("c2", _unit([0.0, 1.0, 0.0, 0.0]), "Blog"),
        _embedded_record("c3", _unit([0.0, 0.0, 1.0, 0.0]), "Admissions"),
    ]
    embedded_path.write_bytes(b"\n".join(orjson.dumps(r) for r in records))

    client = get_qdrant_client(in_memory=True)
    result = ingest_embeddings_to_qdrant(client, embedded_path, collection_name="t")
    assert result["ingested"] == 3
    assert result["collection_count"] == 3


def test_ingestion_is_idempotent(tmp_path: Path) -> None:
    embedded_path = tmp_path / "embedded.jsonl"
    record = _embedded_record("c1", _unit([1.0, 0.0, 0.0, 0.0]), "Admissions")
    embedded_path.write_bytes(orjson.dumps(record))

    client = get_qdrant_client(in_memory=True)
    ingest_embeddings_to_qdrant(client, embedded_path, collection_name="t")
    ingest_embeddings_to_qdrant(client, embedded_path, collection_name="t")
    assert collection_count(client, "t") == 1


def test_search_with_intent_filter(tmp_path: Path) -> None:
    embedded_path = tmp_path / "embedded.jsonl"
    records = [
        _embedded_record("c1", _unit([1.0, 0.0, 0.0, 0.0]), "Admissions"),
        _embedded_record("c2", _unit([0.95, 0.05, 0.0, 0.0]), "Blog"),
    ]
    embedded_path.write_bytes(b"\n".join(orjson.dumps(r) for r in records))

    client = get_qdrant_client(in_memory=True)
    ingest_embeddings_to_qdrant(client, embedded_path, collection_name="t")

    flt = build_intent_filter(allowed_page_types=["Admissions"])
    hits = qdrant_search(
        client,
        query_vector=_unit([1.0, 0.0, 0.0, 0.0]),
        collection_name="t",
        query_filter=flt,
        limit=5,
    )
    assert len(hits) == 1
    assert hits[0].payload["page_type"] == "Admissions"


def test_search_excludes_reusable_components(tmp_path: Path) -> None:
    embedded_path = tmp_path / "embedded.jsonl"
    records = [
        _embedded_record("c1", _unit([1.0, 0.0, 0.0, 0.0]), "Admissions", is_reusable_component=True),
        _embedded_record("c2", _unit([0.9, 0.1, 0.0, 0.0]), "Admissions"),
    ]
    embedded_path.write_bytes(b"\n".join(orjson.dumps(r) for r in records))

    client = get_qdrant_client(in_memory=True)
    ingest_embeddings_to_qdrant(client, embedded_path, collection_name="t")

    flt = build_intent_filter(allowed_page_types=["Admissions"])
    hits = qdrant_search(client, query_vector=_unit([1.0, 0.0, 0.0, 0.0]), collection_name="t", query_filter=flt, limit=5)
    assert len(hits) == 1
    assert hits[0].payload["chunk_id"] == "c2"
