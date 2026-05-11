from __future__ import annotations

from pathlib import Path
from typing import Iterator

import orjson
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

from backend.vectorstore.collection_manager import collection_count, ensure_collection
from backend.vectorstore.payload_mapper import chunk_to_payload
from backend.vectorstore.qdrant_client import DEFAULT_COLLECTION, chunk_id_to_point_id


def _iter_jsonl(path: Path) -> Iterator[dict]:
    with path.open("rb") as fh:
        for line in fh:
            if line.strip():
                yield orjson.loads(line)


def ingest_embeddings_to_qdrant(
    client: QdrantClient,
    embedded_chunks_path: Path = Path("datasets/embedded_chunks.jsonl"),
    collection_name: str = DEFAULT_COLLECTION,
    batch_size: int = 128,
    recreate: bool = False,
) -> dict[str, int]:
    """Idempotent batch upsert. Stable point IDs via UUID5(chunk_id) — re-running this
    overwrites in place. If recreate=True, the collection is dropped first."""
    if not embedded_chunks_path.exists():
        raise FileNotFoundError(embedded_chunks_path)

    chunks_iter = _iter_jsonl(embedded_chunks_path)
    first = next(chunks_iter, None)
    if first is None:
        return {"ingested": 0, "total": 0}

    vector_size = len(first["embedding"])
    ensure_collection(
        client,
        collection_name=collection_name,
        vector_size=vector_size,
        recreate=recreate,
    )

    batch: list[PointStruct] = []
    ingested = 0
    total = 0

    def _flush() -> None:
        nonlocal ingested
        if not batch:
            return
        client.upsert(collection_name=collection_name, points=batch)
        ingested += len(batch)
        batch.clear()

    def _append(chunk: dict) -> None:
        nonlocal total
        total += 1
        batch.append(
            PointStruct(
                id=chunk_id_to_point_id(chunk["chunk_id"]),
                vector=chunk["embedding"],
                payload=chunk_to_payload(chunk),
            )
        )

    _append(first)
    for chunk in chunks_iter:
        _append(chunk)
        if len(batch) >= batch_size:
            _flush()
    _flush()

    return {
        "ingested": ingested,
        "total": total,
        "collection_count": collection_count(client, collection_name),
    }
