import json

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse

from backend.answering.followup_resolution import resolve_followup_query
from backend.answering.grounded_answering import GroundedAnsweringService
from backend.answering.models.answer import GroundedAnswer
from backend.api.chat_models import AnswerRequest, ChatRequest, ChatResponse
from backend.api.chat_ui import CHAT_UI_HTML
from backend.config.settings import settings
from backend.context.context_builder import build_grounded_context
from backend.context.validators import ContextBuildRequest, GroundedContext
from backend.conversation.memory import ConversationMemory
from backend.conversation.session_manager import ensure_session_id
from backend.ingestion.models.document import IngestionStats
from backend.ingestion.services.ingestion_service import IngestionService
from backend.llm.factory import get_provider
from backend.llm.prompts.grounded_answering import SYSTEM_PROMPT, build_user_message
from backend.llm.streaming import to_sse_line
from backend.llm.validators import LLMMessage, LLMRequest
from backend.reranking.validators import RerankedSearchResponse
from backend.retrieval.bm25_service import BM25RetrievalService
from backend.retrieval.dense_retrieval import DenseRetrievalService
from backend.retrieval.hybrid_retrieval import HybridRetrievalService
from backend.retrieval.inspector_html import INSPECTOR_HTML
from backend.retrieval.models.search import SearchResponse
from backend.retrieval.reranked_retrieval import RerankedRetrievalService
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


_bm25_service: BM25RetrievalService | None = None
_dense_service: DenseRetrievalService | None = None
_hybrid_service: HybridRetrievalService | None = None
_reranked_service: RerankedRetrievalService | None = None
_conversation_memory = ConversationMemory()


def _get_bm25() -> BM25RetrievalService:
    global _bm25_service
    if _bm25_service is None:
        _bm25_service = BM25RetrievalService()
    return _bm25_service


def _get_dense() -> DenseRetrievalService:
    global _dense_service
    if _dense_service is None:
        _dense_service = DenseRetrievalService()
    return _dense_service


def _get_hybrid() -> HybridRetrievalService:
    global _hybrid_service
    if _hybrid_service is None:
        _hybrid_service = HybridRetrievalService(bm25=_get_bm25(), dense=_get_dense())
    return _hybrid_service


def _get_reranked() -> RerankedRetrievalService:
    global _reranked_service
    if _reranked_service is None:
        _reranked_service = RerankedRetrievalService(hybrid=_get_hybrid())
    return _reranked_service


@app.get("/retrieval/search", response_model=SearchResponse)
async def retrieval_search(
    query: str,
    top_k: int = 5,
    include_components: bool = False,
) -> SearchResponse:
    return _get_bm25().search(
        query=query, top_k=top_k, include_components=include_components
    )


@app.get("/retrieval/dense/search", response_model=SearchResponse)
async def retrieval_dense_search(
    query: str,
    top_k: int = 5,
    include_components: bool = False,
) -> SearchResponse:
    return _get_dense().search(
        query=query, top_k=top_k, include_components=include_components
    )


@app.get("/retrieval/hybrid/search", response_model=SearchResponse)
async def retrieval_hybrid_search(
    query: str,
    top_k: int = 5,
    include_components: bool = False,
) -> SearchResponse:
    return _get_hybrid().search(
        query=query, top_k=top_k, include_components=include_components
    )


@app.get("/retrieval/reranked/search", response_model=RerankedSearchResponse)
async def retrieval_reranked_search(
    query: str,
    top_k: int = 5,
    candidate_pool: int = 20,
    include_components: bool = False,
) -> RerankedSearchResponse:
    return _get_reranked().search(
        query=query,
        top_k=top_k,
        candidate_pool=candidate_pool,
        include_components=include_components,
    )


@app.post("/context/build", response_model=GroundedContext)
async def context_build(request: ContextBuildRequest) -> GroundedContext:
    reranked = _get_reranked().search(
        query=request.query,
        top_k=request.top_k,
        candidate_pool=request.candidate_pool,
        include_components=request.include_components,
    )
    return build_grounded_context(
        query=request.query,
        intent=reranked.intent,
        reranked=reranked.results,
        token_budget=request.token_budget,
        min_confidence=request.min_grounding_confidence,
        min_blocks=request.min_blocks,
    )


def _build_grounded_context_for(request: AnswerRequest) -> GroundedContext:
    reranked = _get_reranked().search(
        query=request.query,
        top_k=request.top_k,
        candidate_pool=request.candidate_pool,
        include_components=request.include_components,
    )
    return build_grounded_context(
        query=request.query,
        intent=reranked.intent,
        reranked=reranked.results,
        token_budget=request.token_budget,
    )


def _build_answering_service(request: AnswerRequest) -> GroundedAnsweringService:
    provider = get_provider(request.provider)
    return GroundedAnsweringService(
        provider=provider,
        model=request.model or "",
        run_judge=request.run_judge,
    )


@app.post("/answer", response_model=GroundedAnswer)
async def answer(request: AnswerRequest) -> GroundedAnswer:
    grounded_context = _build_grounded_context_for(request)
    service = _build_answering_service(request)
    return service.answer(request.query, grounded_context)


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    session_id = ensure_session_id(request.session_id)
    state = _conversation_memory.append_user_turn(session_id, request.query)
    provider = get_provider(request.provider)

    resolved_query, was_rewritten = resolve_followup_query(
        provider=provider,
        query=request.query,
        state=state,
        model=request.model or "",
    )

    answer_request = AnswerRequest(
        query=resolved_query,
        top_k=request.top_k,
        candidate_pool=request.candidate_pool,
        token_budget=request.token_budget,
        include_components=request.include_components,
        run_judge=request.run_judge,
        provider=request.provider,
        model=request.model,
    )
    grounded_context = _build_grounded_context_for(answer_request)

    reranked_response = _get_reranked().search(
        query=resolved_query,
        top_k=request.top_k,
        candidate_pool=request.candidate_pool,
        include_components=request.include_components,
    )
    routing_filters = {
        "page_types": list(reranked_response.allowed_page_types),
        "section_types": list(reranked_response.allowed_section_types),
    }

    service = _build_answering_service(answer_request)
    grounded_answer = service.answer(
        query=resolved_query,
        grounded_context=grounded_context,
        rewritten_query=resolved_query if was_rewritten else None,
    )

    state = _conversation_memory.append_assistant_turn(
        session_id=session_id,
        content=grounded_answer.answer,
        citations=[c.chunk_id for c in grounded_answer.citations],
        rewritten_query=resolved_query if was_rewritten else None,
        intent=reranked_response.intent,
        routing_filters=routing_filters,
        used_chunks=grounded_answer.used_chunks,
    )

    return ChatResponse(
        session_id=session_id,
        answer=grounded_answer,
        rewritten_query=resolved_query if was_rewritten else None,
        was_rewritten=was_rewritten,
        conversation_state=state,
    )


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest) -> StreamingResponse:
    """Stream answer tokens via Server-Sent Events. The grounded context is built
    synchronously (retrieval is fast); only the LLM generation is streamed."""
    session_id = ensure_session_id(request.session_id)
    state = _conversation_memory.append_user_turn(session_id, request.query)
    provider = get_provider(request.provider)

    resolved_query, was_rewritten = resolve_followup_query(
        provider=provider, query=request.query, state=state, model=request.model or "",
    )
    answer_request = AnswerRequest(
        query=resolved_query,
        top_k=request.top_k,
        candidate_pool=request.candidate_pool,
        token_budget=request.token_budget,
        include_components=request.include_components,
        run_judge=False,
        provider=request.provider,
        model=request.model,
    )
    grounded_context = _build_grounded_context_for(answer_request)
    user_message = build_user_message(resolved_query, grounded_context.prompt)
    llm_request = LLMRequest(
        system_prompt=SYSTEM_PROMPT,
        messages=[LLMMessage(role="user", content=user_message)],
        model=request.model or provider.default_model,
        temperature=0.0,
        max_tokens=1500,
    )

    def event_stream():
        opening = {
            "session_id": session_id,
            "rewritten_query": resolved_query if was_rewritten else None,
            "was_rewritten": was_rewritten,
            "intent": grounded_context.intent,
            "grounding_confidence": grounded_context.grounding_confidence,
        }
        yield b"event: meta\ndata: " + json.dumps(opening).encode() + b"\n\n"
        if not grounded_context.context_blocks:
            yield b"event: abstain\ndata: " + json.dumps(
                {"reason": "no_context_blocks"}
            ).encode() + b"\n\n"
            return
        try:
            for chunk in provider.stream(llm_request):
                yield to_sse_line(chunk)
        except Exception as exc:
            err_payload = {"error": str(exc)}
            yield b"event: error\ndata: " + json.dumps(err_payload).encode() + b"\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/conversation/{session_id}")
async def conversation_state(session_id: str):
    if session_id not in _conversation_memory:
        return {"session_id": session_id, "exists": False}
    return _conversation_memory.get(session_id)


@app.delete("/conversation/{session_id}")
async def reset_conversation(session_id: str):
    _conversation_memory.reset(session_id)
    return {"session_id": session_id, "reset": True}


@app.get("/chat/ui", response_class=HTMLResponse)
async def chat_ui() -> str:
    return CHAT_UI_HTML


@app.get("/chat/provider")
async def chat_provider() -> dict[str, str]:
    """What provider would `/chat` use right now? Lets the UI display 'gemini' vs 'mock'."""
    provider = get_provider()
    return {"provider": provider.name, "default_model": provider.default_model}


@app.get("/retrieval/inspect", response_class=HTMLResponse)
async def retrieval_inspector() -> str:
    return INSPECTOR_HTML
