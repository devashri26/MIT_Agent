from typing import Any, Iterable

from backend.normalization.boilerplate_registry import (
    ReusableComponentRegistry,
    fingerprint_paragraph,
    split_paragraphs,
)


def build_registry(
    chunks: Iterable[dict[str, Any]],
    min_paragraph_chars: int = 40,
) -> ReusableComponentRegistry:
    """Single corpus pass: fingerprint every paragraph, track which document it appears in."""
    registry = ReusableComponentRegistry()
    for chunk in chunks:
        document_id = chunk.get("document_id", "")
        text = chunk.get("text", "")
        for paragraph in split_paragraphs(text, min_chars=min_paragraph_chars):
            fp = fingerprint_paragraph(paragraph)
            registry.register(fp, document_id, paragraph)
    return registry
