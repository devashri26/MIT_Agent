from __future__ import annotations

import os
import uuid
from pathlib import Path

from qdrant_client import QdrantClient


DEFAULT_COLLECTION = "mitaoe_chunks"
DEFAULT_LOCAL_PATH = Path("datasets/qdrant_storage")


def get_qdrant_client(
    storage_path: Path | None = None,
    url: str | None = None,
    in_memory: bool = False,
) -> QdrantClient:
    """Precedence: in_memory > explicit url > QDRANT_URL env > storage_path (default).

    Local file-based mode persists at storage_path. Tests pass in_memory=True.
    """
    if in_memory:
        return QdrantClient(":memory:")
    resolved_url = url or os.environ.get("QDRANT_URL")
    if resolved_url:
        return QdrantClient(url=resolved_url)
    path = storage_path or DEFAULT_LOCAL_PATH
    path.mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=str(path))


def chunk_id_to_point_id(chunk_id: str) -> str:
    """Stable UUID5 derived from chunk_id. Qdrant requires int or UUID point IDs."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))
