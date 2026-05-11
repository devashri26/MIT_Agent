import regex as re

from backend.normalization.boilerplate_registry import (
    ReusableComponentRegistry,
    fingerprint_paragraph,
    split_paragraphs,
)


COMPONENT_PATTERNS: list[tuple[str, str]] = [
    (r"\bfrequently asked questions\b|\bfaqs?\b|^q\d*[:.]\s|\bq\s*&\s*a\b", "faq_widget"),
    (r"\bapply now\b|\benroll today\b|\badmission inquiries?\b|\bregister now\b", "admissions_cta"),
    (r"\bdownload (the )?brochure\b", "cta_brochure"),
    (r"\bcontact us today\b|\bget in touch\b|\benquire now\b", "cta_contact"),
    (r"\bcopyright\b|\ball rights reserved\b|\bprivacy policy\b|\bterms of (use|service)\b", "footer"),
    (r"\bsitemap\b|\bquick links\b|\buseful links\b", "footer"),
    (r"\btop ranked\b|\bnumber one\b|\baward[- ]winning\b|\bnirf rank\b", "promotional"),
    (r"\bbest (private )?engineering colleges?\b", "promotional"),
    (r"\bchat with us\b|\btalk to our\b|\bchatbot\b|\bask our assistant\b", "chatbot_widget"),
    (r"\blearn more\b|\bclick here\b|\bread more\b", "cta_generic"),
]


def classify_component_type(text: str) -> str | None:
    lower = (text or "").lower()
    for pattern, component_type in COMPONENT_PATTERNS:
        if re.search(pattern, lower):
            return component_type
    return None


def classify_registry(
    registry: ReusableComponentRegistry,
    min_doc_count: int = 5,
) -> dict[str, int]:
    """Assign a component_type to every fingerprint that crossed the reuse threshold.

    Fingerprints without a keyword match still get tagged "boilerplate" so they're
    suppressible. Returns component_type → count.
    """
    counts: dict[str, int] = {}
    for fingerprint in registry.reusable_fingerprints(min_doc_count=min_doc_count):
        text = registry.text_for(fingerprint)
        component_type = classify_component_type(text) or "boilerplate"
        registry.set_component_type(fingerprint, component_type)
        counts[component_type] = counts.get(component_type, 0) + 1
    return counts


def assess_chunk_components(
    chunk_text: str,
    registry: ReusableComponentRegistry,
    min_paragraph_chars: int = 40,
    suppress_threshold: float = 0.5,
) -> tuple[bool, str | None, list[str]]:
    """Per-chunk: is the chunk itself a reusable component, and which component types are present?

    Returns (is_reusable_component, dominant_component_type, sorted_component_types_present).
    A chunk is reusable when ≥ suppress_threshold of its character mass is in classified
    component paragraphs.
    """
    paragraphs = split_paragraphs(chunk_text, min_chars=min_paragraph_chars)
    if not paragraphs:
        return False, None, []

    component_chars = 0
    total_chars = sum(len(paragraph) for paragraph in paragraphs)
    type_counter: dict[str, int] = {}

    for paragraph in paragraphs:
        component_type = registry.get_component_type(fingerprint_paragraph(paragraph))
        if component_type:
            component_chars += len(paragraph)
            type_counter[component_type] = type_counter.get(component_type, 0) + 1

    ratio = component_chars / total_chars if total_chars else 0.0
    if ratio >= suppress_threshold and type_counter:
        dominant = max(type_counter, key=lambda k: type_counter[k])
        return True, dominant, sorted(type_counter.keys())
    return False, None, sorted(type_counter.keys())
