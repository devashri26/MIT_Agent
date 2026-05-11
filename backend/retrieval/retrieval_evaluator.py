from typing import Any

from backend.retrieval.benchmark_metrics import METRIC_KEYS, compute_metrics
from backend.retrieval.bm25_service import BM25RetrievalService
from backend.retrieval.dense_retrieval import DenseRetrievalService
from backend.retrieval.hybrid_retrieval import HybridRetrievalService
from backend.retrieval.models.search import SearchResponse


def _run_queries(
    service,
    queries: list[str],
    top_k: int,
) -> list[SearchResponse]:
    return [service.search(query, top_k=top_k) for query in queries]


def evaluate_all_modes(
    queries: list[str],
    corpus: list[dict[str, Any]],
    bm25_service: BM25RetrievalService,
    dense_service: DenseRetrievalService,
    hybrid_service: HybridRetrievalService,
    top_k: int = 10,
) -> dict[str, Any]:
    """Run the same query set through BM25 / dense / hybrid; compute the standard metric
    bundle for each; report hybrid_gain = hybrid - max(bm25, dense) per metric."""
    bm25_responses = _run_queries(bm25_service, queries, top_k)
    dense_responses = _run_queries(dense_service, queries, top_k)
    hybrid_responses = _run_queries(hybrid_service, queries, top_k)

    bm25_metrics = compute_metrics(bm25_responses, corpus)
    dense_metrics = compute_metrics(dense_responses, corpus)
    hybrid_metrics = compute_metrics(hybrid_responses, corpus)

    bm25_overall = bm25_metrics["overall"]
    dense_overall = dense_metrics["overall"]
    hybrid_overall = hybrid_metrics["overall"]

    hybrid_gain: dict[str, float] = {}
    for key in METRIC_KEYS:
        best_single = max(bm25_overall.get(key, 0.0), dense_overall.get(key, 0.0))
        hybrid_gain[key] = round(hybrid_overall.get(key, 0.0) - best_single, 4)

    return {
        "bm25": bm25_metrics,
        "dense": dense_metrics,
        "hybrid": hybrid_metrics,
        "hybrid_gain_over_best_single": hybrid_gain,
        "notes": (
            "All modes use the routing-alignment proxy and exclude reusable components "
            "from the recall denominator. purity@5 = fraction of top-5 NOT reusable, NOT "
            "contaminated, NOT mixed_topic. hybrid_gain = hybrid - max(bm25, dense) per metric."
        ),
    }
