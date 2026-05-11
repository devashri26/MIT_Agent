DOMAIN_OFF_COMPONENTS: dict[str, set[str]] = {
    "Research": {"admissions_cta", "faq_widget", "cta_brochure", "promotional"},
    "Faculty": {"admissions_cta", "faq_widget", "promotional"},
    "Placements": {"faq_widget", "admissions_cta"},
    "Curriculum": {"admissions_cta", "faq_widget", "promotional"},
    "Facilities": {"admissions_cta", "promotional"},
    "Programs": {"admissions_cta"},
}


PAGE_TYPE_EXPECTED_SECTIONS: dict[str, set[str]] = {
    "Admissions": {"admissions", "eligibility", "fees"},
    "Programs": {"curriculum", "syllabus", "admissions", "eligibility", "overview"},
    "Placements": {"placements", "statistics"},
    "Faculty": {"faculty", "research"},
    "Research": {"research", "faculty"},
    "Facilities": {"facilities", "hostel"},
    "Curriculum": {"curriculum", "syllabus"},
    "Club": {"clubs"},
    "Events": {"events"},
    "Notices": {"events"},
    "Blog": set(),
    "General": set(),
}


def detect_cross_domain_contamination(
    page_type: str,
    component_types_in_chunk: list[str],
    mixed_topic: bool,
    dominant_topics: list[str],
) -> tuple[bool, list[str]]:
    """Combine component fingerprint hits with topic-mismatch signals to flag cross-domain
    contamination. Returns (cross_domain_contamination, contamination_sources)."""
    sources: list[str] = []

    off_domain = DOMAIN_OFF_COMPONENTS.get(page_type, set())
    for component_type in component_types_in_chunk:
        if component_type in off_domain:
            sources.append(component_type)

    if mixed_topic and dominant_topics:
        expected = PAGE_TYPE_EXPECTED_SECTIONS.get(page_type, set())
        for topic in dominant_topics:
            if topic and topic != "general" and topic not in expected:
                sources.append(f"off_topic:{topic}")

    deduped: list[str] = []
    seen: set[str] = set()
    for source in sources:
        if source not in seen:
            deduped.append(source)
            seen.add(source)
    return bool(deduped), deduped
