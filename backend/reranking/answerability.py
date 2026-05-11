from __future__ import annotations

import regex as re


# Boost signals — chunks that contain these patterns are more likely to be answer-bearing.
FAQ_MARKERS = [r"^\s*q\d*[:.]", r"\bq\s*&\s*a\b", r"\bfrequently asked\b"]
LIST_MARKERS = [r"^\s*[•\-\*]\s", r"^\s*\d+[\.\)]\s"]
TABLE_MARKERS = [r"\|.*\|.*\|", r"\bs\.no\b.*\bname\b"]
STATS_MARKERS = [
    r"\b\d+\s*%",
    r"\b\d+(\.\d+)?\s*lpa\b",
    r"₹\s*\d",
    r"\brs\.?\s*\d{4,}",
    # Indian number format in MITAOE fee tables: '1, 52, 173.00' or '22,827'
    r"\b\d{1,3}\s*,\s*\d{2,3}\s*,\s*\d{2,3}\b",
    r"\b\d{4}\s*[-–]\s*\d{4}\b",
    r"\btotal[: ]+\d",
]
FACTUAL_KEYWORDS = [
    r"\beligibility\b",
    r"\bminimum (?:qualification|percentage|score)\b",
    r"\bfee structure\b|\btuition\b|\bcharges?\b",
    r"\bsemester\b|\bcredits?\b|\bcore course\b",
    r"\bprofessor\b|\bph\.d\.?\b|\bqualification\b",
    r"\bpublication\b|\bpatent\b|\bjournal\b",
    r"\bhostel\b|\baccommodation\b",
    r"\bplaced\b|\brecruiters?\b",
    r"\bduration\b|\bintake\b|\bdeadline\b",
]
# Strong indicators that a chunk contains an actual fee TABLE (not just a fee mention).
# These get an extra boost so concrete fee data beats marketing pages about fees.
FEE_TABLE_MARKERS = [
    r"\btuition fees?\b",
    r"\bdevelopment fees?\b",
    r"\buniversity fees?\b",
]

# Downrank signals — marketing fluff and CTA chrome.
CTA_PATTERNS = [
    r"\bapply now\b",
    r"\bregister now\b",
    r"\bclick here\b",
    r"\bread more\b",
    r"\bdownload (?:the )?brochure\b",
]
GENERIC_INTRO_PATTERNS = [
    r"\bwelcome to\b",
    r"\bone of the (?:top|leading|best)\b",
    r"\bworld[- ]class\b",
    r"\bunmatched\b",
    r"\bnurturing\b",
    r"\bjourney\b",
]


def _count_matches(patterns: list[str], text: str, multiline: bool = False) -> int:
    flags = re.MULTILINE if multiline else 0
    return sum(1 for pattern in patterns if re.search(pattern, text, flags))


def compute_answerability_score(text: str, token_count: int | None = None) -> float:
    """Heuristic answerability in [0, 1]. Higher = more likely to contain a direct answer."""
    if not text:
        return 0.0
    lowered = text.lower()
    sample = lowered[:2000]

    boosts = 0.0
    boosts += 0.25 if _count_matches(FAQ_MARKERS, sample, multiline=True) else 0.0
    boosts += min(0.20, 0.07 * _count_matches(LIST_MARKERS, sample, multiline=True))
    boosts += 0.15 if _count_matches(TABLE_MARKERS, sample) else 0.0
    boosts += min(0.25, 0.08 * _count_matches(STATS_MARKERS, sample))
    boosts += min(0.30, 0.07 * _count_matches(FACTUAL_KEYWORDS, sample))
    # Fee-table chunks need extra weight so concrete numbers beat marketing prose.
    boosts += min(0.20, 0.10 * _count_matches(FEE_TABLE_MARKERS, sample))

    penalties = 0.0
    cta_hits = _count_matches(CTA_PATTERNS, sample)
    if token_count and token_count > 0:
        penalties += min(0.30, 0.10 * (cta_hits / max(token_count / 100.0, 1.0)))
    else:
        penalties += min(0.30, 0.05 * cta_hits)
    penalties += min(0.20, 0.05 * _count_matches(GENERIC_INTRO_PATTERNS, sample))

    score = 0.40 + boosts - penalties
    return round(max(0.0, min(1.0, score)), 4)
