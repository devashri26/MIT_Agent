from backend.chunking.chunk_pipeline import ChunkPipeline
from backend.chunking.metadata_enricher import MetadataEnricher


def _document() -> dict:
    return {
        "page_id": "doc1",
        "url": "https://example.edu/admissions",
        "title": "Admissions",
        "department": None,
        "page_type": "Admissions",
        "summary": "Admissions summary",
        "clean_content": "Admissions content",
        "sections": [
            {
                "heading": "Eligibility",
                "content": "Students must complete 10+2 with Physics, Chemistry, and Mathematics. They must submit required documents.",
            },
            {
                "heading": "Frequently Asked Questions",
                "content": "1. What exams are accepted?\nJEE Main and MHT-CET are accepted.\n2. When do admissions begin?\nAdmissions normally begin after entrance results.",
            },
        ],
        "metadata": {
            "headings": ["Admissions", "Eligibility"],
            "validation_issues": [],
            "quality_warnings": ["low_page_type_confidence"],
        },
    }


def test_chunk_pipeline_preserves_section_metadata() -> None:
    chunks, report = ChunkPipeline().chunk_documents([_document()])

    assert report.documents_processed == 1
    assert chunks
    assert chunks[0].document_id == "doc1"
    assert chunks[0].metadata.headings == ["Admissions", "Eligibility"]
    assert "low_page_type_confidence" in chunks[0].metadata.quality_warnings


def test_chunk_pipeline_preserves_faq_pairs() -> None:
    chunks, _ = ChunkPipeline().chunk_documents([_document()])
    faq_chunks = [chunk for chunk in chunks if chunk.content_type == "FAQ"]

    assert faq_chunks
    assert "What exams are accepted?" in faq_chunks[0].text
    assert "JEE Main and MHT-CET" in faq_chunks[0].text


def test_chunk_hash_is_deterministic() -> None:
    text = "Admissions require valid entrance exam scores."

    assert MetadataEnricher.chunk_hash(text) == MetadataEnricher.chunk_hash("Admissions   require valid entrance exam scores.")


def test_chunk_pipeline_rejects_useless_fragments() -> None:
    document = _document()
    document["sections"] = [{"heading": "Numbers", "content": "4"}]

    chunks, report = ChunkPipeline().chunk_documents([document])

    assert chunks == []
    assert report.rejected_chunks == 1


def test_chunk_pipeline_merges_tiny_adjacent_chunks() -> None:
    pipeline = ChunkPipeline()

    merged, count = pipeline._merge_tiny_chunks(["What is MCA eligibility?", "Candidates need a qualifying degree."])

    assert count == 1
    assert len(merged) == 1
    assert "qualifying degree" in merged[0]


def test_chunk_pipeline_contextualizes_isolated_tiny_chunks() -> None:
    document = _document()
    document["sections"] = [{"heading": "ARIIA", "content": "ED Cell\nARIIA"}]

    chunks, _ = ChunkPipeline().chunk_documents([document])

    assert chunks
    assert chunks[0].text.startswith("Title: Admissions")
    assert "URL: https://example.edu/admissions" in chunks[0].text
