import hashlib
from collections import defaultdict

import regex as re


def fingerprint_paragraph(text: str) -> str:
    """Lowercase + collapse-whitespace → SHA1 fingerprint. Stable across runs."""
    normalized = re.sub(r"\s+", " ", (text or "").strip().lower())
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()


def split_paragraphs(text: str, min_chars: int = 40) -> list[str]:
    """Split chunk text into paragraphs of at least min_chars.

    Tries blank-line split first; falls back to single-newline split for chunks that
    were re-flowed during cleaning.
    """
    if not text:
        return []
    paragraphs = re.split(r"\n\s*\n", text)
    if len(paragraphs) <= 1:
        paragraphs = re.split(r"\n", text)
    return [p.strip() for p in paragraphs if len(p.strip()) >= min_chars]


class ReusableComponentRegistry:
    """Tracks paragraph fingerprints, the documents each appears in, and component types
    assigned after classification."""

    def __init__(self) -> None:
        self._fingerprint_docs: defaultdict[str, set[str]] = defaultdict(set)
        self._fingerprint_text: dict[str, str] = {}
        self._fingerprint_type: dict[str, str] = {}

    def register(self, fingerprint: str, document_id: str, raw_text: str) -> None:
        self._fingerprint_docs[fingerprint].add(document_id)
        if fingerprint not in self._fingerprint_text:
            self._fingerprint_text[fingerprint] = raw_text

    def document_count(self, fingerprint: str) -> int:
        return len(self._fingerprint_docs.get(fingerprint, set()))

    def reusable_fingerprints(self, min_doc_count: int = 5) -> list[str]:
        return [
            fp
            for fp, docs in self._fingerprint_docs.items()
            if len(docs) >= min_doc_count
        ]

    def set_component_type(self, fingerprint: str, component_type: str) -> None:
        self._fingerprint_type[fingerprint] = component_type

    def get_component_type(self, fingerprint: str) -> str | None:
        return self._fingerprint_type.get(fingerprint)

    def text_for(self, fingerprint: str) -> str:
        return self._fingerprint_text.get(fingerprint, "")
