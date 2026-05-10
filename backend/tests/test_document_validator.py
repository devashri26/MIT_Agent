from backend.ingestion.models.document import DocumentSection, WebsiteDocument
from backend.ingestion.validators.document_validator import DocumentValidator


def test_document_validator_rejects_short_content() -> None:
    document = WebsiteDocument(
        page_id="abc",
        url="https://example.edu/page",
        title="Valid Title",
        department="Computer Engineering",
        page_type="General",
        summary="short",
        sections=[DocumentSection(heading="Overview", content="short")],
        clean_content="short",
        metadata={},
    )

    issues = DocumentValidator().validate(document)

    assert any(issue.field == "clean_content" and issue.severity == "warning" for issue in issues)
