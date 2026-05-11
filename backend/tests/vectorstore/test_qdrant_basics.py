from qdrant_client.models import PointStruct

from backend.vectorstore.collection_manager import collection_count, ensure_collection
from backend.vectorstore.payload_mapper import PAYLOAD_FIELDS, chunk_to_payload
from backend.vectorstore.qdrant_client import chunk_id_to_point_id, get_qdrant_client


def test_in_memory_client_creates_collection() -> None:
    client = get_qdrant_client(in_memory=True)
    ensure_collection(client, collection_name="test", vector_size=384)
    assert client.collection_exists("test")
    assert collection_count(client, "test") == 0


def test_ensure_collection_recreate_drops_data() -> None:
    client = get_qdrant_client(in_memory=True)
    ensure_collection(client, collection_name="test", vector_size=4)
    client.upsert(
        collection_name="test",
        points=[PointStruct(id=chunk_id_to_point_id("a"), vector=[0.1, 0.2, 0.3, 0.4], payload={"x": 1})],
    )
    assert collection_count(client, "test") == 1

    ensure_collection(client, collection_name="test", vector_size=4, recreate=True)
    assert collection_count(client, "test") == 0


def test_chunk_id_to_point_id_stable() -> None:
    a = chunk_id_to_point_id("abc:0000:hash")
    b = chunk_id_to_point_id("abc:0000:hash")
    c = chunk_id_to_point_id("abc:0001:hash")
    assert a == b
    assert a != c


def test_payload_includes_required_fields() -> None:
    chunk = {
        "chunk_id": "c1",
        "page_type": "Admissions",
        "section_type": "eligibility",
        "section_path": ["Admissions", "Eligibility"],
        "retrieval_priority": 0.95,
        "quality_flags": [],
        "is_reusable_component": False,
        "cross_domain_contamination": False,
        "text": "MCA eligibility...",
    }
    payload = chunk_to_payload(chunk)
    assert payload["chunk_id"] == "c1"
    assert payload["page_type"] == "Admissions"
    assert payload["section_path"] == ["Admissions", "Eligibility"]
    assert payload["retrieval_priority"] == 0.95
    assert payload["text"] == "MCA eligibility..."
    for field in ["chunk_id", "page_type", "section_type", "section_path"]:
        assert field in PAYLOAD_FIELDS
