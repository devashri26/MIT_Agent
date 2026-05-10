from pathlib import Path

import orjson

from backend.retrieval.bm25_service import BM25RetrievalService


def test_bm25_retrieval_returns_relevant_chunk() -> None:
    chunks = [
        {
            "chunk_id": "1",
            "document_id": "doc1",
            "url": "https://example.edu/mca",
            "title": "MCA Admissions",
            "department": None,
            "page_type": "Admissions",
            "section_heading": "Eligibility",
            "chunk_index": 0,
            "text": "MCA eligibility requires a relevant undergraduate degree and entrance examination score.",
            "token_count": 14,
            "content_type": "GENERAL",
            "quality_score": 1.0,
            "chunk_hash": "abc",
            "metadata": {"quality_warnings": [], "validation_issues": []},
        },
        {
            "chunk_id": "2",
            "document_id": "doc2",
            "url": "https://example.edu/hostel",
            "title": "Hostel",
            "department": None,
            "page_type": "General",
            "section_heading": "Facilities",
            "chunk_index": 0,
            "text": "Hostel facilities include mess, rooms, and supervision.",
            "token_count": 10,
            "content_type": "GENERAL",
            "quality_score": 1.0,
            "chunk_hash": "def",
            "metadata": {"quality_warnings": [], "validation_issues": []},
        },
    ]
    path = Path("datasets/test_bm25_chunks.jsonl")
    path.write_bytes(b"\n".join(orjson.dumps(chunk) for chunk in chunks))

    response = BM25RetrievalService(path).search("What is MCA eligibility?", top_k=1)

    assert response.results[0].chunk_id == "1"
