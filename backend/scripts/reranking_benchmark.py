from pathlib import Path
from statistics import mean

import orjson

from backend.context.context_builder import build_grounded_context
from backend.retrieval.benchmark_metrics import compute_metrics
from backend.retrieval.bm25_service import BM25RetrievalService
from backend.retrieval.dense_retrieval import DenseRetrievalService
from backend.retrieval.hybrid_retrieval import HybridRetrievalService
from backend.retrieval.reranked_retrieval import RerankedRetrievalService
from backend.scripts.bm25_benchmark import BENCHMARK_QUERIES


def main() -> None:
    bm25 = BM25RetrievalService()
    dense = DenseRetrievalService()
    hybrid = HybridRetrievalService(bm25=bm25, dense=dense)
    reranked = RerankedRetrievalService(hybrid=hybrid, candidate_pool=20)

    hybrid_responses = [hybrid.search(q, top_k=10) for q in BENCHMARK_QUERIES]
    reranked_responses = [reranked.search(q, top_k=10, candidate_pool=20) for q in BENCHMARK_QUERIES]

    hybrid_metrics = compute_metrics(hybrid_responses, bm25.chunks)
    reranked_metrics = compute_metrics(reranked_responses, bm25.chunks)

    rerank_extras = []
    for response in reranked_responses:
        kept = response.results
        rejected_dup = sum(1 for r in response.rejected if r.rejection_reason and r.rejection_reason.startswith("duplicate_of"))
        rejected_div = sum(1 for r in response.rejected if r.rejection_reason and "section_type_saturated" in r.rejection_reason)
        rejected_doc = sum(1 for r in response.rejected if r.rejection_reason and "document_saturated" in r.rejection_reason)
        rerank_extras.append({
            "query": response.query,
            "candidate_pool": response.candidate_pool,
            "kept": len(kept),
            "rejected_total": len(response.rejected),
            "rejected_duplicate": rejected_dup,
            "rejected_diversity_section": rejected_div,
            "rejected_diversity_document": rejected_doc,
            "avg_rerank_score": round(mean([r.rerank_score for r in kept]) if kept else 0.0, 4),
            "avg_answerability": round(mean([r.answerability_score for r in kept]) if kept else 0.0, 4),
            "avg_final_relevance": round(mean([r.final_relevance for r in kept]) if kept else 0.0, 4),
            "distinct_section_types": len({r.section_type for r in kept}),
        })

    reranking_report = {
        "hybrid": hybrid_metrics,
        "reranked": reranked_metrics,
        "reranked_extras": rerank_extras,
        "overall_diffs": {
            key: round(
                reranked_metrics["overall"].get(key, 0.0) - hybrid_metrics["overall"].get(key, 0.0),
                4,
            )
            for key in ["recall@5", "recall@10", "mrr", "hit_rate", "precision@3", "purity@5"]
        },
        "notes": (
            "Rerank operates on the top-20 hybrid candidates per query. The routing-alignment "
            "proxy is unchanged from earlier phases; rerank's value shows up most in purity@5 "
            "and avg_answerability, not in routing metrics."
        ),
    }
    reranking_path = Path("reports/reranking_report.json")
    reranking_path.parent.mkdir(parents=True, exist_ok=True)
    reranking_path.write_bytes(orjson.dumps(reranking_report, option=orjson.OPT_INDENT_2))

    per_query_context: list[dict] = []
    grounding_confidences: list[float] = []
    citation_coverage: list[float] = []
    diversity_counts: list[int] = []
    token_usage: list[int] = []
    for response in reranked_responses:
        ctx = build_grounded_context(
            query=response.query,
            intent=response.intent,
            reranked=response.results,
            token_budget=2000,
        )
        grounding_confidences.append(ctx.grounding_confidence)
        diversity_counts.append(ctx.distinct_section_types)
        token_usage.append(ctx.total_tokens)
        with_section = sum(1 for block in ctx.context_blocks if block.section_path)
        coverage = with_section / max(len(ctx.context_blocks), 1)
        citation_coverage.append(coverage)
        per_query_context.append({
            "query": ctx.query,
            "intent": ctx.intent,
            "context_blocks": len(ctx.context_blocks),
            "distinct_section_types": ctx.distinct_section_types,
            "distinct_documents": ctx.distinct_documents,
            "grounding_confidence": ctx.grounding_confidence,
            "grounding_warnings": ctx.grounding_warnings,
            "total_tokens": ctx.total_tokens,
            "citation_coverage": round(coverage, 4),
        })

    context_quality_report = {
        "queries": len(per_query_context),
        "overall": {
            "avg_grounding_confidence": round(mean(grounding_confidences), 4),
            "avg_citation_coverage": round(mean(citation_coverage), 4),
            "avg_distinct_section_types": round(mean(diversity_counts), 4),
            "avg_context_tokens": round(mean(token_usage), 2),
            "fraction_low_confidence": round(
                sum(1 for c in grounding_confidences if c < 0.5) / max(len(grounding_confidences), 1),
                4,
            ),
        },
        "per_query": per_query_context,
        "notes": "Context assembled from top-5 reranked candidates with 2000-token budget.",
    }
    context_path = Path("reports/context_quality_report.json")
    context_path.write_bytes(orjson.dumps(context_quality_report, option=orjson.OPT_INDENT_2))

    print(f"Wrote {reranking_path}")
    print(f"Wrote {context_path}")
    print(f"Hybrid overall  : {hybrid_metrics['overall']}")
    print(f"Reranked overall: {reranked_metrics['overall']}")
    print(f"Diffs           : {reranking_report['overall_diffs']}")
    print(f"Context summary : {context_quality_report['overall']}")


if __name__ == "__main__":
    main()
