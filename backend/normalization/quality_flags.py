import regex as re


CTA_PATTERNS = [
    r"\bapply now\b",
    r"\bregister now\b",
    r"\bclick here\b",
    r"\bcontact us\b",
    r"\bread more\b",
    r"\bsubmit\b",
    r"\bdownload\b",
    r"\benquire\b",
]

BOILERPLATE_PATTERNS = [
    r"\bcopyright\b",
    r"\ball rights reserved\b",
    r"\bprivacy policy\b",
    r"\bterms of (use|service)\b",
    r"\bcookie policy\b",
    r"\bsitemap\b",
]

EVENT_PATTERNS = [
    r"\b\d{1,2}\s*(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s*\d{2,4}\b",
    r"\bvenue[: ]",
    r"\brsvp\b",
    r"\bregistration deadline\b",
    r"\bscheduled (on|for)\b",
]


def compute_quality_flags(
    text: str,
    token_count: int,
    page_type_confidence: float,
    page_type: str,
    quality_score: float,
) -> list[str]:
    flags: list[str] = []
    text_lower = (text or "").lower()

    if token_count < 60:
        flags.append("low_content")
    elif token_count < 150 and quality_score < 0.5:
        flags.append("thin_content")

    if page_type_confidence < 0.6:
        flags.append("weak_classification")

    cta_hits = sum(1 for pattern in CTA_PATTERNS if re.search(pattern, text_lower))
    normalized = cta_hits / max(token_count / 100.0, 1.0)
    if normalized >= 1.5:
        flags.append("cta_heavy")

    boilerplate_hits = sum(1 for pattern in BOILERPLATE_PATTERNS if re.search(pattern, text_lower))
    if boilerplate_hits >= 2:
        flags.append("boilerplate_heavy")

    event_hits = sum(1 for pattern in EVENT_PATTERNS if re.search(pattern, text_lower))
    if page_type in {"Events", "Notices"} or event_hits >= 3:
        flags.append("event_page")

    return flags
