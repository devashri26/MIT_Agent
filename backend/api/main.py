import json

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse

from backend.config.settings import settings
from backend.ingestion.models.document import IngestionStats
from backend.ingestion.services.ingestion_service import IngestionService
from backend.retrieval.bm25_service import BM25RetrievalService
from backend.retrieval.inspector_html import INSPECTOR_HTML
from backend.retrieval.models.search import SearchResponse
from backend.utils.logging import configure_logging

configure_logging()

app = FastAPI(title="College AI Assistant Backend", version="0.1.0")


@app.post("/ingest", response_model=IngestionStats)
async def ingest(file: UploadFile = File(...)) -> IngestionStats:
    service = IngestionService()
    file.file.seek(0)
    return service.ingest(file.file)


@app.get("/ingestion/report")
async def ingestion_report() -> dict[str, object]:
    report_path = settings.reports_dir / "ingestion_report.json"
    if not report_path.exists():
        return {"message": "No ingestion report has been generated yet."}
    return json.loads(report_path.read_text(encoding="utf-8"))


@app.get("/retrieval/search", response_model=SearchResponse)
async def retrieval_search(query: str, top_k: int = 5) -> SearchResponse:
    return BM25RetrievalService().search(query=query, top_k=top_k)


@app.get("/retrieval/inspect", response_class=HTMLResponse)
async def retrieval_inspector() -> str:
    return INSPECTOR_HTML
