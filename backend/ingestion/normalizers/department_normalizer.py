import regex as re
from rapidfuzz import process


ALLOWED_DEPARTMENTS = [
    "Computer Engineering",
    "Mechanical",
    "Civil",
    "Chemical",
    "Information Technology",
    "Electronics",
    "AI & ML",
    "Data Science",
]


class DepartmentNormalizer:
    _ALIASES: dict[str, str] = {
        "computer": "Computer Engineering",
        "computer engineering": "Computer Engineering",
        "school of computer engineering": "Computer Engineering",
        "sce": "Computer Engineering",
        "mechanical": "Mechanical",
        "mechanical engineering": "Mechanical",
        "civil": "Civil",
        "civil engineering": "Civil",
        "chemical": "Chemical",
        "chemical engineering": "Chemical",
        "information technology": "Information Technology",
        "it": "Information Technology",
        "electronics": "Electronics",
        "electronics engineering": "Electronics",
        "e&tc": "Electronics",
        "entc": "Electronics",
        "etc": "Electronics",
        "ai ml": "AI & ML",
        "artificial intelligence and machine learning": "AI & ML",
        "artificial intelligence machine learning": "AI & ML",
        "data science": "Data Science",
    }

    _NOISE_PATTERN = re.compile(
        r"\b(dr|mrs|mr|ms|prof|professor|assistant professor|associate professor|hod|dean|head|department|school|of|faculty)\b",
        re.IGNORECASE,
    )

    def normalize(self, value: str | None) -> str | None:
        if not value:
            return None

        cleaned = self._clean(value)
        if not cleaned:
            return None

        for alias, department in self._ALIASES.items():
            if re.search(rf"\b{re.escape(alias)}\b", cleaned):
                return department

        match = process.extractOne(cleaned, self._ALIASES.keys(), score_cutoff=88)
        if match:
            return self._ALIASES[match[0]]
        return None

    def infer(self, values: list[str]) -> str | None:
        for value in values:
            normalized = self.normalize(value)
            if normalized:
                return normalized
        return None

    def _clean(self, value: str) -> str:
        value = value.replace("&", " and ")
        value = re.sub(r"[^a-zA-Z0-9+ ]+", " ", value)
        value = self._NOISE_PATTERN.sub(" ", value)
        value = re.sub(r"\b[a-z]\b", " ", value, flags=re.IGNORECASE)
        return re.sub(r"\s+", " ", value).strip().lower()

