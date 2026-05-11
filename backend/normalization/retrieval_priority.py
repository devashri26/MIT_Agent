PAGE_TYPE_PRIORITY: dict[str, float] = {
    "Admissions": 1.00,
    "Programs": 1.00,
    "Curriculum": 0.95,
    "Placements": 0.90,
    "Faculty": 0.75,
    "Facilities": 0.70,
    "Research": 0.70,
    "Blog": 0.40,
    "Club": 0.35,
    "General": 0.30,
    "Events": 0.25,
    "Notices": 0.15,
}


FLAG_PENALTIES: dict[str, float] = {
    "duplicate": 0.40,
    "non_canonical": 0.20,
    "low_content": 0.20,
    "thin_content": 0.10,
    "cta_heavy": 0.15,
    "boilerplate_heavy": 0.10,
    "event_page": 0.10,
    "weak_classification": 0.10,
    "reusable_component": 0.50,
    "cross_domain_contamination": 0.25,
    "mixed_topic": 0.15,
}


def compute_retrieval_priority(
    page_type: str,
    quality_flags: list[str],
    page_type_confidence: float,
    quality_score: float,
) -> float:
    base = PAGE_TYPE_PRIORITY.get(page_type, 0.30)
    confidence_factor = 0.7 + 0.3 * max(0.0, min(page_type_confidence, 1.0))
    quality_factor = 0.7 + 0.3 * max(0.0, min(quality_score, 1.0))
    penalty = sum(FLAG_PENALTIES.get(flag, 0.0) for flag in quality_flags)
    score = base * confidence_factor * quality_factor - penalty
    return round(max(0.0, min(score, 1.0)), 3)
