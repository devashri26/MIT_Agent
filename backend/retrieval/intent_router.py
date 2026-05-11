import regex as re
from pydantic import BaseModel, Field


ALLOWED_INTENTS = [
    "eligibility_query",
    "fees_query",
    "placement_query",
    "faculty_query",
    "hostel_query",
    "curriculum_query",
    "club_query",
    "event_query",
    "research_query",
    "general_query",
]


class IntentRoute(BaseModel):
    intent: str
    allowed_page_types: list[str] = Field(default_factory=list)
    allowed_section_types: list[str] = Field(default_factory=list)
    matched_terms: list[str] = Field(default_factory=list)


_INTENT_RULES: list[tuple[str, str, list[str], list[str]]] = [
    (
        r"\beligibility\b|\bcriteria\b|\bqualif(y|ication)\b|\brequirements?\b|\bminimum\b",
        "eligibility_query",
        ["Admissions", "Programs"],
        ["eligibility", "admissions"],
    ),
    (
        r"\bfees?\b|\btuition\b|\bcost\b|\bcharges?\b|\bscholarships?\b",
        "fees_query",
        ["Admissions", "Programs"],
        ["fees", "admissions"],
    ),
    (
        r"\bplacements?\b|\brecruiters?\b|\bpackage\b|\blpa\b|\bctc\b|\bhighest package\b|\bsalary\b",
        "placement_query",
        ["Placements"],
        ["placements", "statistics"],
    ),
    (
        r"\bclubs?\b|\bsociet(y|ies)\b|\bcommittees?\b|\bieee\b|\bnss\b|\bstudent branch\b",
        "club_query",
        ["Club"],
        ["clubs"],
    ),
    (
        r"\bfaculty\b|\bprofessor\b|\bhod\b|\bcoordinator\b|\bteaching staff\b",
        "faculty_query",
        ["Faculty"],
        ["faculty"],
    ),
    (
        r"\bhostel\b|\baccommodation\b|\bresidential\b|\blibrary\b|\blab(oratory|s)?\b|\bfacilities\b|\binfrastructure\b|\bsports\b",
        "hostel_query",
        ["Facilities"],
        ["hostel", "facilities"],
    ),
    (
        r"\bcurriculum\b|\bsyllabus\b|\bcourse structure\b|\bscheme\b|\bsemester\b|\bcredits?\b|\belectives?\b",
        "curriculum_query",
        ["Curriculum", "Programs"],
        ["curriculum", "syllabus"],
    ),
    (
        r"\bevents?\b|\bfest\b|\bworkshops?\b|\bsymposi(um|a)\b|\bcompetitions?\b|\bnotice\b|\bcircular\b",
        "event_query",
        ["Events", "Notices"],
        ["events"],
    ),
    (
        r"\bresearch\b|\bpublications?\b|\bpatents?\b|\bphd\b|\bjournal\b|\bpaper\b",
        "research_query",
        ["Research", "Faculty"],
        ["research"],
    ),
]


_ALL_HIGH_VALUE_PAGE_TYPES = [
    "Admissions",
    "Programs",
    "Curriculum",
    "Placements",
    "Faculty",
    "Facilities",
    "Research",
    "Club",
]


class IntentRouter:
    """Deterministic query intent classifier. First matching rule wins; falls back to general_query."""

    def route(self, query: str) -> IntentRoute:
        query_lower = (query or "").lower()
        for pattern, intent, page_types, section_types in _INTENT_RULES:
            matched = re.findall(pattern, query_lower)
            if matched:
                return IntentRoute(
                    intent=intent,
                    allowed_page_types=page_types,
                    allowed_section_types=section_types,
                    matched_terms=sorted({m if isinstance(m, str) else m[0] for m in matched if m}),
                )
        return IntentRoute(
            intent="general_query",
            allowed_page_types=_ALL_HIGH_VALUE_PAGE_TYPES,
            allowed_section_types=[],
            matched_terms=[],
        )
