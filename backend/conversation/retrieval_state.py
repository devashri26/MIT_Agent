from __future__ import annotations

import regex as re


PROGRAM_PATTERNS: list[tuple[str, str]] = [
    (r"\bmca\b|\bmaster of computer applications\b", "MCA"),
    (r"\bbtech\b|\bb\.tech\b|\bbachelor of technology\b", "BTech"),
    (r"\bmtech\b|\bm\.tech\b|\bmaster of technology\b", "MTech"),
    (r"\bphd\b|\bdoctorate\b", "PhD"),
]


DEPARTMENT_PATTERNS: list[tuple[str, str]] = [
    (r"\bcomputer engineering\b|\bcs(e)?\b", "Computer Engineering"),
    (r"\bmechanical( engineering)?\b", "Mechanical Engineering"),
    (r"\bcivil( engineering)?\b", "Civil Engineering"),
    (r"\bchemical( engineering)?\b", "Chemical Engineering"),
    (r"\binformation technology\b|\bit dept\b|\bit department\b", "Information Technology"),
    (r"\belectronics( engineering)?\b|\be&tc\b|\bentc\b", "Electronics"),
    (r"\bai/ml\b|\bartificial intelligence\b|\bai ml\b", "AI/ML"),
    (r"\bdata science\b", "Data Science"),
]


def extract_entities(text: str) -> dict[str, str]:
    """Best-effort entity extraction from a query. Used to maintain conversation state so
    follow-up queries inherit the active program/department."""
    found: dict[str, str] = {}
    if not text:
        return found
    lower = text.lower()
    for pattern, label in PROGRAM_PATTERNS:
        if re.search(pattern, lower):
            found["program"] = label
            break
    for pattern, label in DEPARTMENT_PATTERNS:
        if re.search(pattern, lower):
            found["department"] = label
            break
    return found
