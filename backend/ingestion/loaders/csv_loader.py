import logging
from collections.abc import Iterator
from pathlib import Path
from typing import IO, Any

import pandas as pd

from backend.ingestion.models.document import CsvLoadEvent, RawWebsiteRow

logger = logging.getLogger(__name__)


class CsvLoader:
    def __init__(self, chunk_size: int = 1_000) -> None:
        self.chunk_size = chunk_size
        self.bad_line_count = 0

    def iter_rows(self, source: str | Path | IO[bytes] | IO[str]) -> Iterator[CsvLoadEvent]:
        reader = pd.read_csv(
            source,
            chunksize=self.chunk_size,
            dtype=str,
            keep_default_na=False,
            on_bad_lines=self._handle_bad_line,
            engine="python",
        )

        row_number = 1
        for chunk in reader:
            for _, record in chunk.iterrows():
                row_number += 1
                payload = self._record_to_dict(record.to_dict())
                try:
                    raw_row = self._parse_row(payload, row_number)
                except ValueError as exc:
                    logger.warning(
                        "failed_csv_row",
                        extra={"extra": {"row_number": row_number, "error": str(exc)}},
                    )
                    yield CsvLoadEvent(row_number=row_number, error=str(exc))
                    continue
                yield CsvLoadEvent(row=raw_row, row_number=row_number)

    def _handle_bad_line(self, bad_line: list[str]) -> None:
        self.bad_line_count += 1
        logger.warning(
            "malformed_csv_line",
            extra={"extra": {"bad_line_count": self.bad_line_count, "field_count": len(bad_line)}},
        )
        return None

    def _parse_row(self, record: dict[str, str], row_number: int) -> RawWebsiteRow:
        url = self._first_value(record, ["url", "htmlUrl", "metadata/canonicalUrl", "crawl/loadedUrl"])
        html = self._first_value(record, ["html", "raw_html", "body_html"])
        text = self._first_value(record, ["text", "markdown", "content"])
        title = self._first_value(record, ["title", "metadata/title", "metadata/jsonLd/0/headline"])

        if not url:
            raise ValueError("missing source url")
        if not html and not text:
            raise ValueError("missing html/text content")

        metadata = {
            "csv_title": record.get("title", "").strip(),
            "metadata_title": record.get("metadata/title", "").strip(),
            "og_title": self._open_graph_value(record, "og:title"),
            "description": record.get("metadata/description", "").strip(),
            "canonical_url": record.get("metadata/canonicalUrl", "").strip(),
        }
        return RawWebsiteRow(row_number=row_number, url=url, title=title, html=html, text=text, metadata=metadata)

    @staticmethod
    def _record_to_dict(record: dict[str, Any]) -> dict[str, str]:
        return {str(key): "" if value is None else str(value) for key, value in record.items()}

    @staticmethod
    def _first_value(record: dict[str, str], names: list[str]) -> str:
        for name in names:
            value = record.get(name, "").strip()
            if value:
                return value
        return ""

    @staticmethod
    def _open_graph_value(record: dict[str, str], property_name: str) -> str:
        for index in range(50):
            prop = record.get(f"metadata/openGraph/{index}/property", "").strip().lower()
            if prop == property_name:
                return record.get(f"metadata/openGraph/{index}/content", "").strip()
        return ""
