from io import StringIO
import json
from pathlib import Path

from backend.ingestion.loaders.csv_loader import CsvLoader
from backend.ingestion.services.ingestion_service import IngestionService


def test_ingestion_service_exports_documents() -> None:
    csv_data = StringIO(
        "url,title,html\n"
        'https://example.edu/admissions,Admissions,"<main><h1>Admissions</h1><p>Apply now for undergraduate engineering programs. Eligibility details, required documents, admission schedule, counselling process, and fee instructions are published for applicants.</p><p>The admissions office publishes counselling rounds, institute level seats, scholarship guidance, required certificates, contact numbers, and important dates for students and parents.</p></main>"\n'
        'https://example.edu/admissions#copy,Admissions Copy,"<main><h1>Admissions</h1><p>Apply now for undergraduate engineering programs. Eligibility details, required documents, admission schedule, counselling process, and fee instructions are published for applicants.</p><p>The admissions office publishes counselling rounds, institute level seats, scholarship guidance, required certificates, contact numbers, and important dates for students and parents.</p></main>"\n'
        ",Broken,<p>No URL</p>\n"
    )
    output_path = Path("datasets/test_processed_documents.json")
    ndjson_output_path = Path("datasets/test_processed_documents.ndjson")
    reports_dir = Path("reports/test")
    service = IngestionService(loader=CsvLoader(chunk_size=1))

    stats = service.ingest(csv_data, output_path=output_path, ndjson_output_path=ndjson_output_path, reports_dir=reports_dir)
    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert stats.processed == 1
    assert stats.duplicates_removed == 1
    assert stats.failed_rows == 1
    assert payload[0]["url"] == "https://example.edu/admissions"
    assert payload[0]["page_type"] == "Admissions"
    assert payload[0]["sections"]
    assert payload[0]["summary"]
    assert payload[0]["clean_content"].startswith(payload[0]["summary"])
    assert payload[0]["metadata"]["extraction_version"] == "v2"
    assert "quality_warnings" in payload[0]["metadata"]
