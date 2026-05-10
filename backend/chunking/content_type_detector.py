import regex as re


class ContentTypeDetector:
    TYPES = {"FAQ", "FACULTY_PROFILE", "CURRICULUM", "PLACEMENT_STATS", "BLOG", "NOTICE", "EVENT", "GENERAL"}

    def detect(self, page_type: str, heading: str, text: str) -> str:
        evidence = f"{page_type} {heading} {text[:1200]}".lower()
        if self._looks_like_faq(evidence):
            return "FAQ"
        if page_type == "Blog":
            return "BLOG"
        if page_type == "Events":
            return "EVENT"
        if page_type == "Faculty" or re.search(r"\b(professor|qualification|designation|specialization)\b", evidence):
            return "FACULTY_PROFILE"
        if page_type in {"Curriculum", "CourseStructure"} or re.search(r"\b(semester|curriculum|syllabus|course structure)\b", evidence):
            return "CURRICULUM"
        if page_type == "Placements" or re.search(r"\b(placement|package|recruiter|lpa)\b", evidence):
            return "PLACEMENT_STATS"
        if re.search(r"\b(notice|circular|announcement)\b", evidence):
            return "NOTICE"
        return "GENERAL"

    @staticmethod
    def _looks_like_faq(evidence: str) -> bool:
        return bool(re.search(r"\b(faq|frequently asked questions|q\.|question|answer|ans\.)\b", evidence))

