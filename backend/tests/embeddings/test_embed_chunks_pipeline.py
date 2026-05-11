from pathlib import Path

import orjson

from backend.embeddings.embed_chunks import run_embedding


def _normalized_chunk(
    chunk_id: str,
    text: str,
    *,
    is_reusable_component: bool = False,
    cross_domain_contamination: bool = False,
    quality_flags: list[str] | None = None,
) -> dict:
    return {
        "chunk_id": chunk_id,
        "document_id": f"doc_{chunk_id}",
        "url": "https://example.edu/x",
        "canonical_url": "https://example.edu/x",
        "title": "Example",
        "department": None,
        "section_heading": "Overview",
        "chunk_index": 0,
        "text": text,
        "token_count": len(text.split()),
        "content_type": "GENERAL",
        "quality_score": 1.0,
        "chunk_hash": f"hash_{chunk_id}",
        "page_type": "Admissions",
        "page_type_confidence": 0.95,
        "section_type": "admissions",
        "retrieval_priority": 0.9,
        "quality_flags": quality_flags or [],
        "is_canonical": True,
        "section_path": ["Admissions"],
        "is_reusable_component": is_reusable_component,
        "component_type": None,
        "mixed_topic": False,
        "dominant_topics": [],
        "cross_domain_contamination": cross_domain_contamination,
        "contamination_sources": [],
        "embedding_eligible": False,
        "metadata": {},
    }


def test_embed_pipeline_skips_ineligible(tmp_path: Path, fake_model) -> None:
    chunks_path = tmp_path / "normalized.jsonl"
    output_path = tmp_path / "embedded.jsonl"
    cache_path = tmp_path / "cache.npz"
    manifest_path = tmp_path / "manifest.json"

    chunks = [
        _normalized_chunk("eligible1", "MCA eligibility requires a relevant degree."),
        _normalized_chunk("reuse1", "Apply Now!", is_reusable_component=True),
        _normalized_chunk("contam1", "Some content.", cross_domain_contamination=True),
        _normalized_chunk("cta1", "Click Here", quality_flags=["cta_heavy"]),
        _normalized_chunk("boiler1", "Footer text", quality_flags=["boilerplate_heavy"]),
        _normalized_chunk("eligible2", "MCA admissions process is competitive."),
    ]
    chunks_path.write_bytes(b"\n".join(orjson.dumps(c) for c in chunks))

    manifest = run_embedding(
        normalized_chunks_path=chunks_path,
        output_path=output_path,
        cache_path=cache_path,
        manifest_path=manifest_path,
        model=fake_model,
    )

    assert manifest.total_embedded == 2
    assert manifest.skipped_reusable_components == 1
    assert manifest.skipped_contaminated == 1
    assert manifest.skipped_cta_heavy == 1
    assert manifest.skipped_boilerplate_heavy == 1

    embedded_records = [
        orjson.loads(line)
        for line in output_path.read_bytes().splitlines()
        if line.strip()
    ]
    assert len(embedded_records) == 2
    assert all(len(r["embedding"]) == 384 for r in embedded_records)
    assert all(r["embedding_model"] == "fake/test-embedder" for r in embedded_records)
    assert {r["chunk_id"] for r in embedded_records} == {"eligible1", "eligible2"}


def test_embed_pipeline_uses_cache_on_rerun(tmp_path: Path, fake_model) -> None:
    chunks_path = tmp_path / "normalized.jsonl"
    output_path = tmp_path / "embedded.jsonl"
    cache_path = tmp_path / "cache.npz"
    manifest_path = tmp_path / "manifest.json"

    chunks_path.write_bytes(
        orjson.dumps(_normalized_chunk("c1", "MCA eligibility text"))
    )

    first = run_embedding(
        normalized_chunks_path=chunks_path,
        output_path=output_path,
        cache_path=cache_path,
        manifest_path=manifest_path,
        model=fake_model,
    )
    assert first.cache_hits == 0

    second = run_embedding(
        normalized_chunks_path=chunks_path,
        output_path=output_path,
        cache_path=cache_path,
        manifest_path=manifest_path,
        model=fake_model,
    )
    assert second.cache_hits == 1
