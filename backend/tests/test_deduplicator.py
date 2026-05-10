from backend.ingestion.services.deduplicator import Deduplicator


def test_deduplicator_detects_url_duplicates() -> None:
    deduplicator = Deduplicator()

    assert deduplicator.is_duplicate("https://example.edu/page/", "First content") == (False, None, None)
    assert deduplicator.is_duplicate("https://example.edu/page#section", "Different content") == (
        True,
        "url",
        "https://example.edu/page/",
    )


def test_deduplicator_detects_content_duplicates() -> None:
    deduplicator = Deduplicator()

    assert deduplicator.is_duplicate("https://example.edu/a", "Same content") == (False, None, None)
    assert deduplicator.is_duplicate("https://example.edu/b", " same   CONTENT ") == (
        True,
        "content_hash",
        "https://example.edu/a",
    )


def test_deduplicator_keeps_near_duplicates() -> None:
    deduplicator = Deduplicator()

    first = "MIT Academy of Engineering Pune admissions contact details"
    second = "MIT Academy of Engineering Pune statutory compliance contact details"

    assert deduplicator.is_duplicate("https://example.edu/podcast", first) == (False, None, None)
    assert deduplicator.is_duplicate("https://example.edu/statutory-compliance", second) == (False, None, None)
