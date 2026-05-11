import regex as re


PAGE_TYPE_LABELS: dict[str, str] = {
    "Admissions": "Admissions",
    "Programs": "Programs",
    "Faculty": "Faculty",
    "Placements": "Placements",
    "Club": "Student Clubs",
    "Events": "Events",
    "Curriculum": "Curriculum",
    "Research": "Research",
    "Facilities": "Facilities",
    "Notices": "Notices",
    "Blog": "Blog",
    "General": "",
}


SECTION_TYPE_LABELS: dict[str, str] = {
    "eligibility": "Eligibility",
    "fees": "Fees",
    "placements": "Placements",
    "faculty": "Faculty Profiles",
    "research": "Research",
    "facilities": "Facilities",
    "curriculum": "Curriculum",
    "admissions": "Admissions",
    "hostel": "Hostel",
    "internships": "Internships",
    "clubs": "Clubs",
    "events": "Events",
    "contact": "Contact",
    "faq": "FAQ",
    "overview": "",
    "statistics": "Statistics",
    "syllabus": "Syllabus",
    "general": "",
}


def humanize_page_type(page_type: str) -> str:
    return PAGE_TYPE_LABELS.get(page_type or "", page_type or "")


def humanize_section_type(section_type: str) -> str:
    return SECTION_TYPE_LABELS.get(section_type or "", section_type or "")


def clean_document_title(title: str) -> str:
    """Strip brand-name boilerplate and pick the most informative title segment.

    'MCA Admissions | MIT Academy of Engineering' → 'MCA Admissions'
    'MITAOE | Notice Board' → 'Notice Board'
    'Computer Engineering Course Structure' → unchanged
    """
    if not title:
        return ""
    parts = [p.strip() for p in re.split(r"\s*\|\s*", title) if p.strip()]
    if not parts:
        return title.strip()
    if len(parts) == 1:
        return parts[0]
    first = parts[0]
    if re.fullmatch(r"[A-Z]{4,}", first) or "mitaoe" in first.lower() or "mit academy" in first.lower():
        return parts[1] if len(parts) > 1 else first
    return first
