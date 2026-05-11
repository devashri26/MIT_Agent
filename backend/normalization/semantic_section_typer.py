import regex as re

from backend.normalization.validators import ALLOWED_SECTION_TYPES


HEADING_PATTERNS: list[tuple[str, str]] = [
    (r"\beligibility\b|\bminimum qualification\b|\badmission criteria\b", "eligibility"),
    (r"\bfee structure\b|\bfees?\b|\btuition\b|\bcharges?\b", "fees"),
    (r"\bplacements?\b|\brecruiters?\b|\bpackage\b|\bplaced students?\b", "placements"),
    (r"\bfaculty\b|\bprofessor\b|\bteaching staff\b", "faculty"),
    (r"\bresearch\b|\bpublications?\b|\bpatents?\b", "research"),
    (r"\bfacilit(y|ies)\b|\binfrastructure\b|\blibrary\b|\blab(oratory|s)?\b", "facilities"),
    (r"\bcurriculum\b|\bcourse structure\b|\bscheme\b", "curriculum"),
    (r"\bsyllabus\b", "syllabus"),
    (r"\badmissions?\b|\bhow to apply\b|\bapplication process\b", "admissions"),
    (r"\bhostel\b|\baccommodation\b|\bresidential\b", "hostel"),
    (r"\binternships?\b|\bindustry training\b", "internships"),
    (r"\bclubs?\b|\bsociet(y|ies)\b|\bcommittees?\b", "clubs"),
    (r"\bevents?\b|\bworkshops?\b|\bfests?\b|\bsymposi(um|a)\b", "events"),
    (r"\bcontact\b|\baddress\b|\benquir(y|ies)\b|\bget in touch\b", "contact"),
    (r"\bfaqs?\b|\bfrequently asked\b", "faq"),
    (r"\boverview\b|\babout\b|\bintroduction\b", "overview"),
    (r"\bstatistics\b|\bnumbers?\b|\byear[- ]wise\b", "statistics"),
]

FAQ_MARKERS = [r"^\s*q\d*[:.]\s", r"\bq\s*&\s*a\b", r"frequently asked"]
STATS_MARKERS = [r"\b\d+\s*%", r"\b\d+(\.\d+)?\s*lpa\b", r"₹\s*\d", r"\btotal[: ]+\d"]
BULLET_MARKERS = [r"^\s*[•\-\*]\s", r"^\s*\d+[\.\)]\s"]
TABLE_MARKERS = [r"\|.*\|.*\|", r"<table", r"\bs\.no\b.*\bname\b"]


PAGE_TYPE_FALLBACK: dict[str, str] = {
    "Admissions": "admissions",
    "Placements": "placements",
    "Faculty": "faculty",
    "Research": "research",
    "Facilities": "facilities",
    "Curriculum": "curriculum",
    "Programs": "overview",
    "Club": "clubs",
    "Events": "events",
    "Notices": "events",
    "Blog": "general",
    "General": "general",
}


class SemanticSectionTyper:
    """Maps a chunk to one of ALLOWED_SECTION_TYPES via heading > content > page-type fallback."""

    def type_section(
        self,
        section_heading: str,
        headings: list[str],
        text: str,
        page_type: str,
    ) -> str:
        heading_value = " ".join([section_heading or "", *headings[:5]]).lower()
        for pattern, section_type in HEADING_PATTERNS:
            if re.search(pattern, heading_value):
                return section_type

        text_sample = (text or "")[:1500].lower()

        if any(re.search(p, text_sample, re.MULTILINE) for p in FAQ_MARKERS):
            return "faq"

        stats_hits = sum(1 for p in STATS_MARKERS if re.search(p, text_sample))
        if stats_hits >= 2:
            return "statistics"

        if self._has_structured_content(text_sample):
            for pattern, section_type in HEADING_PATTERNS:
                if re.search(pattern, text_sample):
                    return section_type

        fallback = PAGE_TYPE_FALLBACK.get(page_type, "general")
        return fallback if fallback in ALLOWED_SECTION_TYPES else "general"

    @staticmethod
    def _has_structured_content(text: str) -> bool:
        if any(re.search(p, text, re.MULTILINE) for p in BULLET_MARKERS):
            return True
        if any(re.search(p, text) for p in TABLE_MARKERS):
            return True
        return False
