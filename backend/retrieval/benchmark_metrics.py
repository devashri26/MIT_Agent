from collections import Counter, defaultdict
from typing import Any

from backend.retrieval.models.search import SearchResponse


METRIC_KEYS = ["recall@5", "recall@10", "mrr", "hit_rate", "precision@3", "purity@5"]


def compute_metrics(
    responses: list[SearchResponse],
    corpus: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compute retrieval metrics using the routing-alignment proxy.

    A result is treated as 'relevant' if its page_type is in the query's allowed_page_types.
    Recall denominators use the corpus-wide count of chunks with allowed page_types. This is
    a routing-quality signal, not a true relevance metric (no ground-truth labels).
    """
    page_type_corpus_counts: Counter[str] = Counter(
        chunk.get("page_type", "")
        for chunk in corpus
        if not chunk.get("is_reusable_component")
    )

    per_query: list[dict[str, Any]] = []
    per_intent: defaultdict[str, list[dict[str, float]]] = defaultdict(list)

    for response in responses:
        allowed = set(response.allowed_page_types)
        relevant_total = sum(page_type_corpus_counts[pt] for pt in allowed)

        top10 = response.results[:10]
        top5 = response.results[:5]
        top3 = response.results[:3]

        matches_top10 = sum(1 for r in top10 if r.page_type in allowed)
        matches_top5 = sum(1 for r in top5 if r.page_type in allowed)
        matches_top3 = sum(1 for r in top3 if r.page_type in allowed)

        first_match_rank = next((r.rank for r in top10 if r.page_type in allowed), 0)
        rr = 1.0 / first_match_rank if first_match_rank else 0.0

        recall_at_5 = (matches_top5 / relevant_total) if relevant_total else 0.0
        recall_at_10 = (matches_top10 / relevant_total) if relevant_total else 0.0
        precision_at_3 = matches_top3 / max(len(top3), 1)
        hit_rate = 1.0 if matches_top10 > 0 else 0.0

        pure_top5 = sum(
            1 for r in top5
            if not r.is_reusable_component
            and not r.cross_domain_contamination
            and not r.mixed_topic
        )
        purity_at_5 = pure_top5 / max(len(top5), 1)

        query_metrics = {
            "query": response.query,
            "intent": response.intent,
            "filter_fallback_used": response.filter_fallback_used,
            "components_excluded": response.components_excluded,
            "relevant_total_corpus": relevant_total,
            "matches_top10": matches_top10,
            "recall@5": round(recall_at_5, 4),
            "recall@10": round(recall_at_10, 4),
            "mrr": round(rr, 4),
            "hit_rate": hit_rate,
            "precision@3": round(precision_at_3, 4),
            "purity@5": round(purity_at_5, 4),
        }
        per_query.append(query_metrics)
        per_intent[response.intent].append({
            "recall@5": recall_at_5,
            "recall@10": recall_at_10,
            "mrr": rr,
            "hit_rate": hit_rate,
            "precision@3": precision_at_3,
            "purity@5": purity_at_5,
        })

    per_intent_summary: dict[str, dict[str, float]] = {}
    for intent, rows in per_intent.items():
        per_intent_summary[intent] = {
            "n_queries": len(rows),
            **{
                key: round(sum(row[key] for row in rows) / len(rows), 4)
                for key in METRIC_KEYS
            },
        }

    n = len(per_query)
    if n:
        overall: dict[str, Any] = {
            "n_queries": n,
            **{
                key: round(sum(q[key] for q in per_query) / n, 4)
                for key in METRIC_KEYS
            },
        }
    else:
        overall = {"n_queries": 0}

    return {
        "overall": overall,
        "per_intent": per_intent_summary,
        "per_query": per_query,
        "notes": (
            "Routing-alignment proxy: a result is 'relevant' if its page_type is in the "
            "query's allowed_page_types. Recall@K denominator = NON-component corpus chunks "
            "with allowed page_type (components excluded from both numerator and denominator). "
            "purity@5 = fraction of top-5 results that are NOT reusable, NOT contaminated, "
            "NOT mixed_topic. Not a true relevance metric — requires hand-labeled ground truth."
        ),
    }
