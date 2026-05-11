from pathlib import Path

import orjson

from backend.retrieval.bm25_service import BM25RetrievalService
from backend.retrieval.dense_retrieval import DenseRetrievalService
from backend.retrieval.hybrid_retrieval import HybridRetrievalService
from backend.retrieval.retrieval_evaluator import evaluate_all_modes
from backend.scripts.bm25_benchmark import BENCHMARK_QUERIES


def main() -> None:
    bm25 = BM25RetrievalService()
    dense = DenseRetrievalService()
    hybrid = HybridRetrievalService(bm25=bm25, dense=dense)

    metrics = evaluate_all_modes(
        queries=BENCHMARK_QUERIES,
        corpus=bm25.chunks,
        bm25_service=bm25,
        dense_service=dense,
        hybrid_service=hybrid,
        top_k=10,
    )

    report_path = Path("reports/hybrid_retrieval_report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_bytes(orjson.dumps(metrics, option=orjson.OPT_INDENT_2))

    print(f"Wrote report to {report_path}")
    print(f"BM25 overall  : {metrics['bm25']['overall']}")
    print(f"Dense overall : {metrics['dense']['overall']}")
    print(f"Hybrid overall: {metrics['hybrid']['overall']}")
    print(f"Hybrid gain   : {metrics['hybrid_gain_over_best_single']}")


if __name__ == "__main__":
    main()
