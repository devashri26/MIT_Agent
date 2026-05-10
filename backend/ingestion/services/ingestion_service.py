import logging
from collections import Counter
from pathlib import Path
from typing import IO
from urllib.parse import urlparse

from rapidfuzz import fuzz

from backend.config.settings import settings
from backend.ingestion.classifiers.page_type_classifier import PageTypeClassifier
from backend.ingestion.cleaners.html_cleaner import HtmlCleaner
from backend.ingestion.exporters.json_exporter import JsonExporter
from backend.ingestion.extractors.title_extractor import TitleExtractor
from backend.ingestion.loaders.csv_loader import CsvLoader
from backend.ingestion.models.document import (
    DuplicateRecord,
    FailedRowRecord,
    IngestionReport,
    IngestionStats,
    ValidationIssue,
    WebsiteDocument,
)
from backend.ingestion.normalizers.department_normalizer import DepartmentNormalizer
from backend.ingestion.normalizers.text_normalizer import TextNormalizer
from backend.ingestion.parsers.html_parser import HtmlParser
from backend.ingestion.reports.report_writer import ReportWriter
from backend.ingestion.services.deduplicator import Deduplicator
from backend.ingestion.validators.document_validator import DocumentValidator

logger = logging.getLogger(__name__)


class IngestionService:
    def __init__(
        self,
        loader: CsvLoader | None = None,
        cleaner: HtmlCleaner | None = None,
        normalizer: TextNormalizer | None = None,
        parser: HtmlParser | None = None,
        title_extractor: TitleExtractor | None = None,
        department_normalizer: DepartmentNormalizer | None = None,
        page_type_classifier: PageTypeClassifier | None = None,
        validator: DocumentValidator | None = None,
        exporter: JsonExporter | None = None,
        report_writer: ReportWriter | None = None,
    ) -> None:
        self.loader = loader or CsvLoader(chunk_size=settings.csv_chunk_size)
        self.cleaner = cleaner or HtmlCleaner()
        self.normalizer = normalizer or TextNormalizer()
        self.parser = parser or HtmlParser()
        self.title_extractor = title_extractor or TitleExtractor()
        self.department_normalizer = department_normalizer or DepartmentNormalizer()
        self.page_type_classifier = page_type_classifier or PageTypeClassifier()
        self.validator = validator or DocumentValidator()
        self.exporter = exporter or JsonExporter()
        self.report_writer = report_writer or ReportWriter()

    def ingest(
        self,
        source: str | Path | IO[bytes] | IO[str],
        output_path: Path = settings.processed_output_path,
        ndjson_output_path: Path = settings.processed_ndjson_output_path,
        reports_dir: Path = settings.reports_dir,
    ) -> IngestionStats:
        deduplicator = Deduplicator()
        documents: list[WebsiteDocument] = []
        stats = IngestionStats(output_path=str(output_path), report_path=str(reports_dir / "ingestion_report.json"))
        duplicates: list[DuplicateRecord] = []
        failed_rows: list[FailedRowRecord] = []
        validation_summary: dict[str, int] = {}
        page_type_counts: Counter[str] = Counter()
        quality_warning_counts: Counter[str] = Counter()
        boilerplate_removals = 0

        for event in self.loader.iter_rows(source):
            stats.csv_rows += 1
            if event.is_failed or event.row is None:
                stats.failed_rows += 1
                failed_rows.append(FailedRowRecord(row_number=event.row_number, reason=event.error or "failed csv row"))
                continue

            row = event.row
            try:
                clean_html_result = self.cleaner.clean_html_result(row.html) if row.html else None
                cleaned_html = clean_html_result.html if clean_html_result else ""
                boilerplate_removals += clean_html_result.removed_blocks_count if clean_html_result else 0
                cleaned, malformed = self.cleaner.clean(row.html) if row.html else (row.text, False)
                stats.malformed_html += int(malformed)
                clean_content = self.normalizer.normalize(cleaned or row.text)
                if not clean_content:
                    stats.empty_content += 1
                    raise ValueError("empty content after cleaning")

                title_result = self.title_extractor.extract(row)
                title = title_result.title
                if any(issue.field == "title" for issue in title_result.issues):
                    stats.invalid_titles += 1

                headings = self.parser.headings(cleaned_html or row.html)
                sections = self.parser.sections(cleaned_html or row.html, clean_content)
                department = self.department_normalizer.infer([row.url, title, *headings])
                page_type, page_type_confidence, classifier_warnings = self.page_type_classifier.classify(
                    row.url, title, headings, clean_content
                )
                summary = self._summary(clean_content)
                quality_warnings = self._quality_warnings(
                    clean_content=clean_content,
                    page_type=page_type,
                    classifier_warnings=classifier_warnings,
                    multi_article_warning=clean_html_result.multi_article_warning if clean_html_result else False,
                )

                is_duplicate, reason, matched_url = deduplicator.is_duplicate(row.url, clean_content)
                if is_duplicate:
                    stats.duplicates_removed += 1
                    duplicates.append(
                        DuplicateRecord(
                            row_number=row.row_number,
                            url=row.url,
                            reason=reason or "duplicate",
                            matched_url=matched_url,
                        )
                    )
                    logger.info(
                        "duplicate_removed",
                        extra={
                            "extra": {
                                "row_number": row.row_number,
                                "url": row.url,
                                "reason": reason,
                                "matched_url": matched_url,
                            }
                        },
                    )
                    continue

                near_duplicate_url = deduplicator.near_duplicate_match(row.url, clean_content)
                if near_duplicate_url:
                    quality_warnings.append("duplicate_content")

                document = WebsiteDocument(
                    page_id=deduplicator.page_id(row.url, clean_content),
                    url=row.url,
                    title=title,
                    department=department,
                    page_type=page_type,
                    summary=summary,
                    sections=sections,
                    clean_content=clean_content,
                    metadata={
                        "source_url": row.url,
                        "url_path": urlparse(row.url).path,
                        "title_source": title_result.source,
                        "title_url_similarity": title_result.similarity,
                        "content_length": len(clean_content),
                        "section_count": len(sections),
                        "headings": headings,
                        "validation_issues": [],
                        "quality_warnings": quality_warnings,
                        "extraction_version": "v2",
                        "page_type_confidence": page_type_confidence,
                        "removed_boilerplate_blocks_count": clean_html_result.removed_blocks_count if clean_html_result else 0,
                        "near_duplicate_of": near_duplicate_url,
                    },
                )
                validation_issues = [
                    *title_result.issues,
                    *self._content_alignment_issues(row.url, title, clean_content),
                    *self.validator.validate(document),
                ]
                document.metadata["validation_issues"] = [issue.model_dump(mode="json") for issue in validation_issues]
                for issue in validation_issues:
                    if issue.message == "content is below retrieval-quality threshold":
                        quality_warnings.append("low_content")
                    if issue.message == "title has low similarity to extracted content":
                        quality_warnings.append("title_content_mismatch")
                document.metadata["quality_warnings"] = sorted(set(quality_warnings))
                self._count_validation_issues(validation_summary, validation_issues)
                blocking_issues = [issue for issue in validation_issues if issue.severity == "error"]
                if blocking_issues:
                    self._apply_stats_for_validation(stats, blocking_issues)
                    raise ValueError("; ".join(f"{issue.field}: {issue.message}" for issue in blocking_issues))
            except Exception as exc:
                stats.failed_rows += 1
                failed_rows.append(FailedRowRecord(row_number=row.row_number, url=row.url, reason=str(exc)))
                logger.warning(
                    "failed_processing_row",
                    extra={"extra": {"row_number": row.row_number, "url": row.url, "error": str(exc)}},
                )
                continue

            documents.append(document)
            stats.processed += 1
            page_type_counts[document.page_type or "General"] += 1
            quality_warning_counts.update(document.metadata.get("quality_warnings", []))

        self.exporter.export(documents, output_path)
        self.exporter.export_ndjson(documents, ndjson_output_path)
        integrity = self._integrity(stats)
        report = IngestionReport(stats=stats, validation_summary=validation_summary, integrity=integrity)
        validation_report = self._validation_report(
            stats=stats,
            validation_summary=validation_summary,
            page_type_counts=page_type_counts,
            quality_warning_counts=quality_warning_counts,
            boilerplate_removals=boilerplate_removals,
        )
        extraction_stats = {
            "extraction_version": "v2",
            "documents_exported": len(documents),
            "average_sections_per_document": round(
                sum(len(document.sections) for document in documents) / len(documents), 2
            )
            if documents
            else 0,
            "average_content_length": round(
                sum(len(document.clean_content) for document in documents) / len(documents), 2
            )
            if documents
            else 0,
            "boilerplate_removals": boilerplate_removals,
        }
        self.report_writer.write(report, duplicates, failed_rows, reports_dir, validation_report, extraction_stats)
        logger.info(
            "ingestion_metrics",
            extra={
                "extra": {
                    "processed": stats.processed,
                    "duplicates_removed": stats.duplicates_removed,
                    "failed_rows": stats.failed_rows,
                    "skipped_rows": stats.skipped_rows,
                    "malformed_html": stats.malformed_html,
                    "output_path": stats.output_path,
                    "report_path": stats.report_path,
                }
            },
        )
        return stats

    @staticmethod
    def _summary(content: str, max_length: int = 500) -> str:
        paragraphs = [part.strip() for part in content.split("\n") if len(part.strip()) > 40]
        text = " ".join((paragraphs[0] if paragraphs else content).split())
        if len(text) <= max_length:
            return text
        return text[:max_length].rsplit(" ", 1)[0].strip()

    @staticmethod
    def _quality_warnings(
        clean_content: str,
        page_type: str,
        classifier_warnings: list[str],
        multi_article_warning: bool,
    ) -> list[str]:
        warnings = list(classifier_warnings)
        if len(clean_content) < 300:
            warnings.append("low_content")
        if page_type == "DirectoryIndex":
            warnings.append("directory_listing")
        if multi_article_warning:
            warnings.append("possible_multi_article_bleed")
        return sorted(set(warnings))

    @staticmethod
    def _content_alignment_issues(url: str, title: str, content: str) -> list[ValidationIssue]:
        slug = " ".join(Path(urlparse(url).path).stem.replace("-", " ").replace("_", " ").split())
        evidence = f"{title} {content[:1000]}"
        if not slug or not evidence.strip():
            return []
        score = fuzz.token_set_ratio(slug, evidence)
        if score < 25:
            return [
                ValidationIssue(
                    field="title",
                    message="title has low similarity to extracted content",
                    severity="warning",
                )
            ]
        return []

    @staticmethod
    def _count_validation_issues(summary: dict[str, int], issues: list[ValidationIssue]) -> None:
        for issue in issues:
            key = f"{issue.field}_{issue.severity}"
            summary[key] = summary.get(key, 0) + 1

    @staticmethod
    def _apply_stats_for_validation(stats: IngestionStats, issues: list[ValidationIssue]) -> None:
        fields = {issue.field for issue in issues}
        if "clean_content" in fields:
            stats.empty_content += 1
        if "department" in fields:
            stats.invalid_departments += 1
        if "page_type" in fields:
            stats.invalid_page_types += 1

    @staticmethod
    def _integrity(stats: IngestionStats) -> dict[str, object]:
        accounted = stats.processed + stats.duplicates_removed + stats.failed_rows + stats.skipped_rows
        return {
            "csv_rows": stats.csv_rows,
            "accounted_rows": accounted,
            "rows_balance": stats.csv_rows - accounted,
            "is_balanced": stats.csv_rows == accounted,
        }

    @staticmethod
    def _validation_report(
        stats: IngestionStats,
        validation_summary: dict[str, int],
        page_type_counts: Counter[str],
        quality_warning_counts: Counter[str],
        boilerplate_removals: int,
    ) -> dict[str, object]:
        return {
            "total_pages": stats.processed,
            "low_content_pages": quality_warning_counts.get("low_content", 0),
            "duplicate_pages": quality_warning_counts.get("duplicate_content", 0) + stats.duplicates_removed,
            "mismatched_pages": quality_warning_counts.get("title_content_mismatch", 0),
            "boilerplate_removals": boilerplate_removals,
            "extraction_failures": stats.failed_rows,
            "page_type_distribution": dict(sorted(page_type_counts.items())),
            "validation_summary": validation_summary,
            "quality_warning_distribution": dict(sorted(quality_warning_counts.items())),
        }
