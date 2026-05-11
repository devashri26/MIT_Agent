from __future__ import annotations

from qdrant_client import QdrantClient
from qdrant_client.models import Filter

from backend.vectorstore.qdrant_client import DEFAULT_COLLECTION


def qdrant_search(
    client: QdrantClient,
    query_vector: list[float],
    collection_name: str = DEFAULT_COLLECTION,
    query_filter: Filter | None = None,
    limit: int = 20,
) -> list:
    """Returns a list of ScoredPoint (id, score, payload). Empty list when collection
    has no points or no matches survive filtering."""
    response = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        query_filter=query_filter,
        limit=limit,
        with_payload=True,
        with_vectors=False,
    )
    return response.points
