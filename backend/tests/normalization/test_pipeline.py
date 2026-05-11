from pathlib import Path

import orjson

from backend.normalization.pipeline import run_normalization


def _write_chunks(path: Path, chunks: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\n".join(orjson.dumps(c) for c in chunks))


def test_pipeline_enriches_chunks(tmp_path: Path) -> None:
    chunks_path = tmp_path / "chunks.jsonl"
    documents_path = tmp_path / "documents.json"
    output_path = tmp_path / "normalized_chunks.jsonl"
    report_path = tmp_path / "corpus_normalization_report.json"

    documents_path.write_bytes(orjson.dumps([
        {
            "page_id": "doc1",
            "url": "https://mitaoe.ac.in/admissions/mca",
            "sections": [{"heading": "Eligibility", "content": "..."}],
            "clean_content": "MCA eligibility minimum qualification entrance.",
        },
    ]))
    _write_chunks(chunks_path, [
        {
            "chunk_id": "c1",
            "document_id": "doc1",
            "url": "https://mitaoe.ac.in/admissions/mca",
            "title": "MCA Admissions",
            "department": "MCA",
            "page_type": "Admissions",
            "section_heading": "Eligibility",
            "chunk_index": 0,
            "text": "MCA eligibility requires a relevant undergraduate degree.",
            "token_count": 400,
            "content_type": "GENERAL",
            "quality_score": 0.9,
            "chunk_hash": "h1",
            "metadata": {"headings": ["Eligibility"], "quality_warnings": []},
        },
        {
            "chunk_id": "c2",
            "document_id": "doc2",
            "url": "https://mitaoe.ac.in/notice/1",
            "title": "Notice 1",
            "department": None,
            "page_type": "Events",
            "section_heading": "Notice",
            "chunk_index": 0,
            "text": "Notice dated.",
            "token_count": 10,
            "content_type": "GENERAL",
            "quality_score": 0.2,
            "chunk_hash": "h2",
            "metadata": {"headings": [], "quality_warnings": []},
        },
    ])

    report = run_normalization(
        chunks_path=chunks_path,
        documents_path=documents_path,
        output_path=output_path,
        report_path=report_path,
    )

    assert report.total_chunks == 2
    assert report.page_type_distribution.get("Admissions") == 1
    assert report.page_type_distribution.get("Notices") == 1

    lines = [orjson.loads(line) for line in output_path.read_bytes().splitlines() if line.strip()]
    assert len(lines) == 2

    admissions_chunk = next(c for c in lines if c["chunk_id"] == "c1")
    assert admissions_chunk["page_type"] == "Admissions"
    assert admissions_chunk["section_type"] == "eligibility"
    assert admissions_chunk["retrieval_priority"] >= 0.7
    assert admissions_chunk["is_canonical"] is True
    assert "low_content" not in admissions_chunk["quality_flags"]

    notice_chunk = next(c for c in lines if c["chunk_id"] == "c2")
    assert notice_chunk["page_type"] == "Notices"
    assert "low_content" in notice_chunk["quality_flags"]
    assert "event_page" in notice_chunk["quality_flags"]
    assert notice_chunk["retrieval_priority"] < admissions_chunk["retrieval_priority"]


def test_pipeline_marks_duplicate_and_non_canonical(tmp_path: Path) -> None:
    chunks_path = tmp_path / "chunks.jsonl"
    output_path = tmp_path / "normalized_chunks.jsonl"
    report_path = tmp_path / "corpus_normalization_report.json"

    _write_chunks(chunks_path, [
        {
            "chunk_id": "c1",
            "document_id": "doc1",
            "url": "https://mitaoe.ac.in/admissions",
            "title": "Admissions",
            "department": None,
            "page_type": "Admissions",
            "section_heading": "Overview",
            "chunk_index": 0,
            "text": "Admissions information.",
            "token_count": 200,
            "content_type": "GENERAL",
            "quality_score": 1.0,
            "chunk_hash": "samehash",
            "metadata": {"headings": [], "quality_warnings": []},
        },
        {
            "chunk_id": "c2",
            "document_id": "doc2",
            "url": "https://www.mitaoe.ac.in/admissions/",
            "title": "Admissions",
            "department": None,
            "page_type": "Admissions",
            "section_heading": "Overview",
            "chunk_index": 0,
            "text": "Admissions information.",
            "token_count": 200,
            "content_type": "GENERAL",
            "quality_score": 1.0,
            "chunk_hash": "samehash",
            "metadata": {"headings": [], "quality_warnings": []},
        },
    ])

    run_normalization(
        chunks_path=chunks_path,
        documents_path=tmp_path / "nonexistent.json",
        output_path=output_path,
        report_path=report_path,
    )

    lines = [orjson.loads(line) for line in output_path.read_bytes().splitlines() if line.strip()]
    duplicate_chunk = next(c for c in lines if c["chunk_id"] == "c2")
    assert "duplicate" in duplicate_chunk["quality_flags"]
    assert "non_canonical" in duplicate_chunk["quality_flags"]
    assert duplicate_chunk["is_canonical"] is False
