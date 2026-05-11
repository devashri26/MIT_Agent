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


# Paragraphs matching these patterns are ACADEMIC content (fee tables, eligibility
# criteria, faculty bios, curriculum, placement stats) that legitimately appear in many
# program pages by design. Per Phase 3 spec: "NEVER suppress academic information,
# eligibility content, faculty data, curriculum, research content, placement statistics".
PRESERVED_CONTENT_PATTERNS = [
    # Fee tables / tuition (the BTech fee bug that motivated this list)
    r"\btuition fees?\b",
    r"\bdevelopment fees?\b",
    r"\buniversity fees?\b",
    r"\bfee structure\b",
    r"₹\s*\d",
    r"\brs\.?\s*\d{4,}",
    r"\b\d{1,3}\s*,\s*\d{2,3}\s*,\s*\d{2,3}\b",
    # Eligibility / admission criteria
    r"\beligibility (?:criteria|requirements?|for)\b",
    r"\bminimum (?:qualification|percentage|aggregate|marks)\b",
    r"\bentrance exam(?:ination)?\b|\bgate score\b|\bjee main\b|\bmht[- ]cet\b|\bcet score\b",
    # Curriculum / academic structure
    r"\bsemester\s*\d\b|\bcredit hours?\b|\bcore courses?\b|\belective courses?\b",
    r"\bcourse structure\b|\bsyllabus\b|\bscheme\b",
    # Faculty bios
    r"\bph\.?d\.?\b.{0,30}(?:from|in|at)",
    r"\bresearch interests?\s*:",
    r"\bteaching experience\b|\bindustry experience\b",
    # Placement stats
    r"\b\d+(?:\.\d+)?\s*lpa\b",
    r"\bhighest package\b|\baverage package\b|\bplacement (?:statistics|offers?|cell)\b",
    # Hostel / facilities
    r"\bhostel (?:rooms?|fees?|charges?|facilities|accommodation)\b",
]


def is_preserved_content(text: str) -> bool:
    """High-value academic content that must never be flagged as a reusable component
    even when it appears in many documents. The Phase 3 spec explicitly forbids
    suppressing this kind of content."""
    if not text:
        return False
    lower = text.lower()
    return any(re.search(pattern, lower) for pattern in PRESERVED_CONTENT_PATTERNS)


def classify_component_type(text: str) -> str | None:
    if is_preserved_content(text):
        return None
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
        if is_preserved_content(text):
            # Academic content (fee tables, eligibility, faculty bios, etc.) legitimately
            # repeats across program pages — leave it unflagged so retrieval can find it.
            continue
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
    component paragraphs — UNLESS the chunk contains academic content (fees, eligibility,
    faculty bios, etc.) anywhere, in which case it is never marked reusable. The fee-table
    chunks have boilerplate around them ("submit applications to the authorities…") that
    dominates by character mass; we still want the fee data retrievable.
    """
    paragraphs = split_paragraphs(chunk_text, min_chars=min_paragraph_chars)
    if not paragraphs:
        return False, None, []

    has_preserved = any(is_preserved_content(p) for p in paragraphs)

    component_chars = 0
    total_chars = sum(len(paragraph) for paragraph in paragraphs)
    type_counter: dict[str, int] = {}

    for paragraph in paragraphs:
        component_type = registry.get_component_type(fingerprint_paragraph(paragraph))
        if component_type:
            component_chars += len(paragraph)
            type_counter[component_type] = type_counter.get(component_type, 0) + 1

    if has_preserved:
        # Spare the chunk from reusable-component flagging — academic content is present.
        return False, None, sorted(type_counter.keys())

    ratio = component_chars / total_chars if total_chars else 0.0
    if ratio >= suppress_threshold and type_counter:
        dominant = max(type_counter, key=lambda k: type_counter[k])
        return True, dominant, sorted(type_counter.keys())
    return False, None, sorted(type_counter.keys())
