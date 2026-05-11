from pathlib import Path

from backend.vectorstore.ingestion import ingest_embeddings_to_qdrant
from backend.vectorstore.qdrant_client import DEFAULT_COLLECTION, get_qdrant_client


def main() -> None:
    client = get_qdrant_client()
    result = ingest_embeddings_to_qdrant(
        client=client,
        embedded_chunks_path=Path("datasets/embedded_chunks.jsonl"),
        collection_name=DEFAULT_COLLECTION,
        recreate=True,
    )
    print(f"Ingested {result['ingested']} of {result['total']} embedded chunks")
    print(f"Collection '{DEFAULT_COLLECTION}' now has {result['collection_count']} points")


if __name__ == "__main__":
    main()
