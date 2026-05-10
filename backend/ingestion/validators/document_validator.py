from urllib.parse import urlparse

from backend.ingestion.classifiers.page_type_classifier import ALLOWED_PAGE_TYPES
from backend.ingestion.models.document import ValidationIssue, WebsiteDocument
from backend.ingestion.normalizers.department_normalizer import ALLOWED_DEPARTMENTS


class DocumentValidator:
    MIN_CONTENT_LENGTH = 300
    MIN_TITLE_LENGTH = 6

    def validate(self, document: WebsiteDocument) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []

        if not self._valid_url(document.url):
            issues.append(ValidationIssue(field="url", message="url is malformed"))
        if len(document.title.strip()) < self.MIN_TITLE_LENGTH:
            issues.append(ValidationIssue(field="title", message="title is too short"))
        if len(document.clean_content.strip()) < self.MIN_CONTENT_LENGTH:
            issues.append(
                ValidationIssue(field="clean_content", message="content is below retrieval-quality threshold", severity="warning")
            )
        if document.department is not None and document.department not in ALLOWED_DEPARTMENTS:
            issues.append(ValidationIssue(field="department", message="department is not whitelisted"))
        if document.page_type not in ALLOWED_PAGE_TYPES:
            issues.append(ValidationIssue(field="page_type", message="page type is not whitelisted"))
        if not document.sections:
            issues.append(ValidationIssue(field="sections", message="no structured sections extracted", severity="warning"))

        return issues

    @staticmethod
    def _valid_url(url: str) -> bool:
        parsed = urlparse(url)
        return parsed.scheme in {"http", "https"} and bool(parsed.netloc)
