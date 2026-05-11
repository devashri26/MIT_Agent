from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from backend.vectorstore.qdrant_client import DEFAULT_COLLECTION


def ensure_collection(
    client: QdrantClient,
    collection_name: str = DEFAULT_COLLECTION,
    vector_size: int = 384,
    recreate: bool = False,
) -> None:
    """Idempotent collection creation. With recreate=True, drops existing data first."""
    exists = client.collection_exists(collection_name)
    if exists and recreate:
        client.delete_collection(collection_name)
        exists = False
    if not exists:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )


def collection_count(client: QdrantClient, collection_name: str = DEFAULT_COLLECTION) -> int:
    return int(client.count(collection_name=collection_name, exact=True).count)
