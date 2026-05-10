from typing import Any

from pydantic import BaseModel, Field


class DocumentSection(BaseModel):
    heading: str
    content: str


class WebsiteDocument(BaseModel):
    page_id: str
    url: str
    title: str
    department: str | None = None
    page_type: str | None = None
    summary: str
    sections: list[DocumentSection] = Field(default_factory=list)
    clean_content: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class IngestionStats(BaseModel):
    csv_rows: int = 0
    processed: int = 0
    duplicates_removed: int = 0
    failed_rows: int = 0
    skipped_rows: int = 0
    empty_content: int = 0
    invalid_titles: int = 0
    invalid_departments: int = 0
    invalid_page_types: int = 0
    malformed_html: int = 0
    output_path: str | None = None
    report_path: str | None = None


class RawWebsiteRow(BaseModel):
    row_number: int
    url: str
    title: str = ""
    html: str = ""
    text: str = ""
    metadata: dict[str, str] = Field(default_factory=dict)


class CsvLoadEvent(BaseModel):
    row: RawWebsiteRow | None = None
    row_number: int
    error: str | None = None

    @property
    def is_failed(self) -> bool:
        return self.row is None


class ValidationIssue(BaseModel):
    field: str
    message: str
    severity: str = "error"


class DuplicateRecord(BaseModel):
    row_number: int
    url: str
    reason: str
    matched_url: str | None = None


class FailedRowRecord(BaseModel):
    row_number: int
    url: str | None = None
    reason: str


class IngestionReport(BaseModel):
    stats: IngestionStats
    validation_summary: dict[str, int] = Field(default_factory=dict)
    integrity: dict[str, Any] = Field(default_factory=dict)
