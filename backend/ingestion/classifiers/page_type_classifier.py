import regex as re
from urllib.parse import urlparse


ALLOWED_PAGE_TYPES = [
    "Blog",
    "Department",
    "Admissions",
    "Placements",
    "Faculty",
    "Research",
    "Curriculum",
    "CourseStructure",
    "Events",
    "DirectoryIndex",
    "General",
]


class PageTypeClassifier:
    URL_RULES: list[tuple[str, str]] = [
        (r"/\?c=[nmsd];o=[ad]$|/\?c=[nmsd]&o=[ad]$", "DirectoryIndex"),
        (r"/blog/", "Blog"),
        (r"\bblog\b", "Blog"),
        (r"course-structure|course_structure", "CourseStructure"),
        (r"curriculum|syllabus|scheme", "Curriculum"),
        (r"faculty|staff|professor|people", "Faculty"),
        (r"admission|apply|fee", "Admissions"),
        (r"placement|recruiter|career", "Placements"),
        (r"research|publication|patent|project|phd", "Research"),
        (r"event|achievement|notice|circular|announcement|news", "Events"),
        (r"department|school-of|computer-engineering|mechanical|civil|chemical|information-technology|electronics|btech-", "Department"),
    ]
    TEXT_RULES: list[tuple[str, str]] = [
        (r"\bindex of\b", "DirectoryIndex"),
        (r"\bcourse structure\b", "CourseStructure"),
        (r"\bcurriculum\b|\bsyllabus\b", "Curriculum"),
        (r"\bfaculty\b|\bprofessor\b|\bstaff\b", "Faculty"),
        (r"\badmissions?\b|\beligibility\b", "Admissions"),
        (r"\bplacements?\b|\brecruiters?\b", "Placements"),
        (r"\bresearch\b|\bpublications?\b|\bpatents?\b", "Research"),
        (r"\bevents?\b|\bachievements?\b|\bnotice\b|\bcircular\b|\bannouncement\b", "Events"),
        (r"\bdepartment\b|\bschool of\b", "Department"),
    ]

    def classify(self, url: str, title: str, headings: list[str], content: str) -> tuple[str, float, list[str]]:
        parsed = urlparse(url)
        url_value = f"{parsed.path} {parsed.query}".lower()
        is_root_page = parsed.path in {"", "/"}
        if self.is_directory_index(title, content):
            return "DirectoryIndex", 1.0, []

        for pattern, page_type in self.URL_RULES:
            if re.search(pattern, url_value, flags=re.IGNORECASE):
                return page_type, 0.95, []

        title_value = title.lower()
        for pattern, page_type in self.TEXT_RULES:
            if re.search(pattern, title_value, flags=re.IGNORECASE):
                return page_type, 0.8, []

        heading_value = " ".join(headings[:6]).lower()
        for pattern, page_type in self.TEXT_RULES:
            if re.search(pattern, heading_value, flags=re.IGNORECASE):
                return page_type, 0.7, []

        if is_root_page:
            return "General", 0.9, []

        content_value = content[:1_500].lower()
        for pattern, page_type in self.TEXT_RULES:
            if page_type in {"Faculty", "Placements"}:
                continue
            if re.search(pattern, content_value, flags=re.IGNORECASE):
                return page_type, 0.45, ["low_page_type_confidence"]

        return "General", 0.35, ["low_page_type_confidence"]

    @staticmethod
    def is_directory_index(title: str, content: str) -> bool:
        evidence = f"{title}\n{content[:500]}".lower()
        return "index of /" in evidence or bool(re.search(r"\bname\s+last modified\s+size\b", evidence))
