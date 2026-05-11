import regex as re

from backend.normalization.boilerplate_registry import split_paragraphs
from backend.normalization.semantic_section_typer import HEADING_PATTERNS


def detect_paragraph_topic(text: str) -> str:
    """Classify a single paragraph into a section_type using heading-keyword patterns.

    Always runs the keyword patterns (no structured-content gate, unlike the document-level
    SemanticSectionTyper). Falls back to 'general' when no keyword matches.
    """
    if not text:
        return "general"
    text_lower = text[:500].lower()
    for pattern, section_type in HEADING_PATTERNS:
        if re.search(pattern, text_lower):
            return section_type
    return "general"


def detect_mixed_topic(
    chunk_text: str,
    min_paragraph_chars: int = 40,
    topic_threshold: float = 0.3,
) -> tuple[bool, list[str]]:
    """Return (mixed_topic, dominant_topics).

    A chunk is mixed when ≥2 distinct non-general topics each cover ≥ topic_threshold
    of its paragraphs. dominant_topics is sorted by paragraph count descending.
    """
    paragraphs = split_paragraphs(chunk_text, min_chars=min_paragraph_chars)
    if len(paragraphs) < 2:
        return False, []

    counts: dict[str, int] = {}
    for paragraph in paragraphs:
        topic = detect_paragraph_topic(paragraph)
        counts[topic] = counts.get(topic, 0) + 1

    total = len(paragraphs)
    strong_topics = [
        topic
        for topic, n in counts.items()
        if topic != "general" and n / total >= topic_threshold
    ]
    strong_topics.sort(key=lambda t: counts[t], reverse=True)
    return len(strong_topics) >= 2, strong_topics
