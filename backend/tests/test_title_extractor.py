from backend.ingestion.extractors.title_extractor import TitleExtractor
from backend.ingestion.models.document import RawWebsiteRow


def test_title_extractor_uses_deterministic_priority() -> None:
    row = RawWebsiteRow(
        row_number=2,
        url="https://example.edu/admissions",
        title="CSV Title",
        html="<html><head><title>HTML Title</title></head><body><h1>H1 Title</h1></body></html>",
        metadata={"og_title": "OG Title", "csv_title": "CSV Title"},
    )

    result = TitleExtractor().extract(row)

    assert result.title == "OG Title"
    assert result.source == "og:title"


def test_title_extractor_flags_url_mismatch() -> None:
    similarity = TitleExtractor.url_title_similarity(
        "https://example.edu/how-industry-4.0-is-revolutionising-technology-operations",
        "Benefits of Honours and Minor Degrees",
    )

    assert similarity is not None
    assert similarity < TitleExtractor.URL_TITLE_WARNING_THRESHOLD

