from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import orjson

from backend.embeddings.batch_embedder import BatchEmbedder
from backend.embeddings.eligibility import is_embedding_eligible, skip_reason
from backend.embeddings.embedding_cache import EmbeddingCache
from backend.embeddings.embedding_model import EmbeddingModel
from backend.embeddings.validators import EmbeddingManifest


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(path)
    return [orjson.loads(line) for line in path.read_bytes().splitlines() if line.strip()]


def run_embedding(
    normalized_chunks_path: Path = Path("datasets/normalized_chunks.jsonl"),
    output_path: Path = Path("datasets/embedded_chunks.jsonl"),
    cache_path: Path = Path("datasets/embedding_cache.npz"),
    manifest_path: Path = Path("reports/embedding_manifest.json"),
    batch_size: int = 32,
    model: EmbeddingModel | None = None,
) -> EmbeddingManifest:
    chunks = _load_jsonl(normalized_chunks_path)

    eligible: list[dict] = []
    skipped_counts: Counter[str] = Counter()
    for chunk in chunks:
        if is_embedding_eligible(chunk):
            eligible.append(chunk)
        else:
            reason = skip_reason(chunk) or "unknown"
            skipped_counts[reason] += 1

    model = model or EmbeddingModel()
    cache = EmbeddingCache(cache_path=cache_path)
    cache_size_before = len(cache)
    embedder = BatchEmbedder(model=model, cache=cache, batch_size=batch_size)

    hashes = [chunk["chunk_hash"] for chunk in eligible]
    texts = [chunk["text"] for chunk in eligible]
    vectors = embedder.embed(hashes, texts)
    cache.save()

    embedded_at = datetime.now(timezone.utc).isoformat()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as out_file:
        for chunk, vector in zip(eligible, vectors):
            record = {
                **chunk,
                "embedding": vector.tolist(),
                "embedding_dimension": int(vector.shape[0]),
                "embedded_at": embedded_at,
                "embedding_model": model.model_name,
            }
            out_file.write(orjson.dumps(record))
            out_file.write(b"\n")

    manifest = EmbeddingManifest(
        embedding_model=model.model_name,
        embedding_dimension=model.dimension,
        total_eligible_chunks=len(eligible),
        total_embedded=len(eligible),
        skipped_reusable_components=skipped_counts.get("reusable_component", 0),
        skipped_contaminated=skipped_counts.get("cross_domain_contamination", 0),
        skipped_cta_heavy=skipped_counts.get("cta_heavy", 0),
        skipped_boilerplate_heavy=skipped_counts.get("boilerplate_heavy", 0),
        cache_hits=cache_size_before,
        embedded_at=embedded_at,
        output_path=str(output_path),
        skip_reason_distribution=dict(skipped_counts),
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_bytes(
        orjson.dumps(manifest.model_dump(mode="json"), option=orjson.OPT_INDENT_2)
    )
    return manifest


def main() -> None:
    manifest = run_embedding()
    print(f"Embedded {manifest.total_embedded} chunks with {manifest.embedding_model}")
    print(f"  Dimension: {manifest.embedding_dimension}")
    print(f"  Cache hits at start: {manifest.cache_hits}")
    print(f"  Skipped: {manifest.skip_reason_distribution}")


if __name__ == "__main__":
    main()
