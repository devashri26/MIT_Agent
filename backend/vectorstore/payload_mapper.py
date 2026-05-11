from typing import Any


PAYLOAD_FIELDS = [
    "chunk_id",
    "document_id",
    "url",
    "canonical_url",
    "title",
    "department",
    "page_type",
    "page_type_confidence",
    "section_type",
    "section_heading",
    "section_path",
    "retrieval_priority",
    "quality_flags",
    "quality_score",
    "is_reusable_component",
    "component_type",
    "mixed_topic",
    "dominant_topics",
    "cross_domain_contamination",
    "contamination_sources",
    "token_count",
    "content_type",
    "embedding_model",
    "embedded_at",
]


def chunk_to_payload(chunk: dict[str, Any]) -> dict[str, Any]:
    """Project a normalized + embedded chunk into a Qdrant payload.

    Excludes the embedding vector itself (stored separately as the point's vector).
    Includes `text` so hybrid retrieval can reconstruct results without a second lookup.
    """
    payload: dict[str, Any] = {field: chunk.get(field) for field in PAYLOAD_FIELDS}
    payload["text"] = chunk.get("text", "")
    return payload
