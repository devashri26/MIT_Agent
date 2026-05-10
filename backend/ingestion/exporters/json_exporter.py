import json
from pathlib import Path

from backend.ingestion.models.document import WebsiteDocument


class JsonExporter:
    def export(self, documents: list[WebsiteDocument], output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = [document.model_dump(mode="json") for document in documents]
        output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    def export_ndjson(self, documents: list[WebsiteDocument], output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        lines = [document.model_dump_json() for document in documents]
        output_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

