from urllib.parse import urlparse

import regex as re

from backend.normalization.validators import ALLOWED_PAGE_TYPES


URL_RULES: list[tuple[str, str, float]] = [
    (r"/club-|/clubs?/|/student-clubs?", "Club", 0.97),
    (r"/blog/|/blog-|/articles?/", "Blog", 0.95),
    (r"/admission|/apply|/eligibility|/how-to-apply", "Admissions", 0.96),
    (r"/placement|/recruiter|/training-and-placement", "Placements", 0.96),
    (r"/curriculum|/syllabus|/course-structure|/scheme", "Curriculum", 0.95),
    (r"/faculty|/professor|/people|/staff", "Faculty", 0.95),
    (r"/research|/publication|/patent|/phd", "Research", 0.94),
    (r"/hostel|/library|/laborator|/labs?/|/sports|/facilities|/infrastructure", "Facilities", 0.94),
    (r"/notice|/circular|/announcement|/tender", "Notices", 0.93),
    (r"/event|/fest|/cultural|/symposium|/workshop", "Events", 0.92),
    (r"/btech-|/mtech-|/mca-|/be-|/me-|/programs?/|/program-|/courses?/", "Programs", 0.92),
    (
        r"/department|/school-of|/computer-engineering|/mechanical|/civil|/chemical|/information-technology|/electronics",
        "Programs",
        0.85,
    ),
]


TITLE_RULES: list[tuple[str, str, float]] = [
    (r"\bclub\b|\bsociety\b|\bcommittee\b", "Club", 0.85),
    (r"\badmissions?\b|\beligibility\b|\bhow to apply\b", "Admissions", 0.88),
    (r"\bplacements?\b|\brecruiters?\b|\btraining and placement\b", "Placements", 0.88),
    (r"\bcurriculum\b|\bsyllabus\b|\bcourse structure\b|\bscheme\b", "Curriculum", 0.85),
    (r"\bfaculty\b|\bprofessor\b|\bstaff\b|\bdr\.", "Faculty", 0.82),
    (r"\bresearch\b|\bpublications?\b|\bpatents?\b|\bphd\b", "Research", 0.82),
    (r"\bhostel\b|\blibrary\b|\blab(oratory|s)?\b|\bsports\b|\bfacilities\b", "Facilities", 0.82),
    (r"\bnotice\b|\bcircular\b|\bannouncement\b|\btender\b", "Notices", 0.83),
    (r"\bevent\b|\bfest\b|\bsymposium\b|\bworkshop\b", "Events", 0.78),
    (r"\bbtech\b|\bmtech\b|\bmca\b|\bb\.tech\b|\bm\.tech\b|\bprograms?\b", "Programs", 0.78),
    (r"\bblog\b|\barticle\b", "Blog", 0.78),
]


HEADING_RULES: list[tuple[str, str, float]] = [
    (r"\beligibility\b|\badmission process\b|\bminimum qualification\b", "Admissions", 0.7),
    (r"\bplacement statistics\b|\brecruiters?\b|\bpackage\b", "Placements", 0.7),
    (r"\bcurriculum\b|\bsyllabus\b|\bcourse structure\b", "Curriculum", 0.7),
    (r"\bfaculty members?\b|\bteaching staff\b", "Faculty", 0.7),
    (r"\bresearch areas?\b|\bpublications?\b", "Research", 0.7),
    (r"\bhostel\b|\blibrary\b|\bcampus facilities\b|\binfrastructure\b", "Facilities", 0.7),
    (r"\bnotice\b|\bcircular\b", "Notices", 0.65),
    (r"\bevent\b|\bworkshop\b|\bfest\b", "Events", 0.65),
    (r"\bprograms? offered\b|\bdegree programs?\b", "Programs", 0.7),
]


CONTENT_KEYWORDS: dict[str, tuple[list[str], float]] = {
    "Admissions": (["eligibility", "admission", "entrance", "minimum qualification", "apply"], 0.55),
    "Placements": (["placement", "recruiter", "package", "lpa", "ctc", "placed"], 0.55),
    "Curriculum": (["semester", "credits", "core course", "elective", "syllabus"], 0.55),
    "Faculty": (["professor", "ph.d.", "research interests", "qualification"], 0.5),
    "Research": (["publication", "patent", "journal", "conference paper", "doi"], 0.55),
    "Facilities": (["hostel", "library", "laboratory", "auditorium", "infrastructure"], 0.5),
    "Notices": (["notice", "circular", "tender", "dated"], 0.5),
    "Events": (["registration", "rsvp", "scheduled", "venue", "date and time"], 0.45),
    "Club": (["committee", "office bearers", "club activities"], 0.5),
    "Programs": (["b.tech", "m.tech", "mca", "program structure", "course outcomes"], 0.5),
    "Blog": (["read more", "posted by", "comments"], 0.4),
}


class DeterministicPageClassifier:
    """Priority-ordered deterministic page classifier.

    Order: URL → title → heading → content keyword voting → General fallback.
    Returns (page_type, confidence). page_type is always in ALLOWED_PAGE_TYPES.
    """

    CONFIDENCE_FLOOR = 0.6

    def classify(
        self,
        url: str,
        title: str,
        headings: list[str],
        content: str,
    ) -> tuple[str, float]:
        url_value = self._url_string(url)
        title_value = (title or "").lower()
        heading_value = " ".join(headings[:8]).lower()
        content_sample = (content or "")[:2000].lower()

        for pattern, page_type, confidence in URL_RULES:
            if re.search(pattern, url_value):
                return page_type, confidence

        for pattern, page_type, confidence in TITLE_RULES:
            if re.search(pattern, title_value):
                return page_type, confidence

        for pattern, page_type, confidence in HEADING_RULES:
            if re.search(pattern, heading_value):
                return page_type, confidence

        votes: dict[str, int] = {}
        for page_type, (keywords, _base) in CONTENT_KEYWORDS.items():
            hits = sum(1 for kw in keywords if kw in content_sample)
            if hits >= 2:
                votes[page_type] = hits
        if votes:
            winner = max(votes, key=lambda k: votes[k])
            base_conf = CONTENT_KEYWORDS[winner][1]
            confidence = min(0.7, base_conf + 0.05 * (votes[winner] - 2))
            if confidence >= self.CONFIDENCE_FLOOR:
                return winner, confidence

        return "General", 0.3

    @staticmethod
    def _url_string(url: str) -> str:
        parsed = urlparse(url or "")
        return f"{parsed.path} {parsed.query}".lower()


def is_valid_page_type(page_type: str) -> bool:
    return page_type in ALLOWED_PAGE_TYPES
