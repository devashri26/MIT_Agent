from backend.normalization.page_classifier import DeterministicPageClassifier
from backend.normalization.validators import ALLOWED_PAGE_TYPES


def test_url_rules_dominate() -> None:
    classifier = DeterministicPageClassifier()

    assert classifier.classify("https://example.edu/club-ieee", "IEEE", [], "")[0] == "Club"
    assert classifier.classify("https://example.edu/faculty/jane", "Profile", [], "")[0] == "Faculty"
    assert classifier.classify("https://example.edu/placement-statistics", "Placements", [], "")[0] == "Placements"
    assert classifier.classify("https://example.edu/admission-process", "How to apply", [], "")[0] == "Admissions"
    assert classifier.classify("https://example.edu/curriculum-btech", "Scheme", [], "")[0] == "Curriculum"


def test_url_beats_title_when_both_match() -> None:
    classifier = DeterministicPageClassifier()
    page_type, _ = classifier.classify(
        "https://example.edu/blog/exam-tips",
        "Admissions Eligibility Criteria",
        [],
        "",
    )
    assert page_type == "Blog"


def test_title_used_when_url_is_neutral() -> None:
    classifier = DeterministicPageClassifier()
    page_type, confidence = classifier.classify(
        "https://example.edu/about",
        "MCA Admissions Eligibility",
        [],
        "",
    )
    assert page_type == "Admissions"
    assert confidence >= 0.6


def test_falls_back_to_general_with_low_confidence() -> None:
    classifier = DeterministicPageClassifier()
    page_type, confidence = classifier.classify("https://example.edu/", "Home", [], "Welcome to MITAOE.")
    assert page_type == "General"
    assert confidence < 0.6


def test_returned_type_always_allowed() -> None:
    classifier = DeterministicPageClassifier()
    for url in [
        "https://example.edu/blog/x",
        "https://example.edu/faculty",
        "https://example.edu/random/path",
        "",
    ]:
        page_type, _ = classifier.classify(url, "", [], "")
        assert page_type in ALLOWED_PAGE_TYPES


def test_department_url_maps_to_programs() -> None:
    classifier = DeterministicPageClassifier()
    assert classifier.classify("https://example.edu/computer-engineering", "", [], "")[0] == "Programs"
    assert classifier.classify("https://example.edu/btech-it", "", [], "")[0] == "Programs"


def test_keyword_voting_fallback() -> None:
    classifier = DeterministicPageClassifier()
    content = "Eligibility for admission requires minimum qualification. Apply via the entrance exam."
    page_type, confidence = classifier.classify("https://example.edu/info", "Information", [], content)
    assert page_type == "Admissions"
    assert confidence >= 0.6
