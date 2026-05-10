import json
from pathlib import Path

from backend.ingestion.models.document import DuplicateRecord, FailedRowRecord, IngestionReport


class ReportWriter:
    def write(
        self,
        report: IngestionReport,
        duplicates: list[DuplicateRecord],
        failed_rows: list[FailedRowRecord],
        reports_dir: Path,
        validation_report: dict[str, object] | None = None,
        extraction_stats: dict[str, object] | None = None,
    ) -> None:
        reports_dir.mkdir(parents=True, exist_ok=True)
        self._write_json(reports_dir / "ingestion_report.json", report.model_dump(mode="json"))
        self._write_json(reports_dir / "duplicate_report.json", [item.model_dump(mode="json") for item in duplicates])
        self._write_json(reports_dir / "failed_rows.json", [item.model_dump(mode="json") for item in failed_rows])
        if validation_report is not None:
            self._write_json(reports_dir / "validation_report.json", validation_report)
        if extraction_stats is not None:
            self._write_json(reports_dir / "extraction_stats.json", extraction_stats)

    @staticmethod
    def _write_json(path: Path, payload: object) -> None:
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
