from pathlib import Path

import orjson

from backend.answering.grounded_answering import GroundedAnsweringService
from backend.context.context_builder import build_grounded_context
from backend.evaluation.answer_quality import evaluate_answer_quality
from backend.evaluation.grounding_metrics import compute_grounding_metrics
from backend.evaluation.hallucination_metrics import compute_hallucination_metrics
from backend.llm.factory import get_provider
from backend.retrieval.reranked_retrieval import RerankedRetrievalService


QA_SET_PATH = Path("backend/evaluation/qa_sets/grounded_qa.json")


def main() -> None:
    qa_set = orjson.loads(QA_SET_PATH.read_bytes())

    reranked = RerankedRetrievalService()
    provider = get_provider()
    answering = GroundedAnsweringService(provider=provider, run_judge=True)

    records: list[dict] = []
    for qa in qa_set:
        rerank_response = reranked.search(qa["query"], top_k=5, candidate_pool=20)
        grounded_context = build_grounded_context(
            query=qa["query"],
            intent=rerank_response.intent,
            reranked=rerank_response.results,
            token_budget=2000,
        )
        answer = answering.answer(qa["query"], grounded_context)
        record = evaluate_answer_quality(answer, qa)
        record["intent"] = rerank_response.intent
        record["n_context_blocks"] = len(grounded_context.context_blocks)
        records.append(record)
        print(
            f"[{qa['id']}] {qa['query'][:50]}  abstained={record['abstained']}  "
            f"correct_abstain={record['abstention_correct']}  citations={record['n_citations']}  "
            f"hallucination_risk={record['hallucination_risk']}"
        )

    report = {
        "provider": provider.name,
        "default_model": provider.default_model,
        "per_query": records,
        "grounding": compute_grounding_metrics(records),
        "hallucination": compute_hallucination_metrics(records),
    }
    report_path = Path("reports/answer_quality_report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_bytes(orjson.dumps(report, option=orjson.OPT_INDENT_2))
    print(f"\nWrote {report_path}")
    print(f"Grounding: {report['grounding']}")
    print(f"Hallucination: {report['hallucination']}")


if __name__ == "__main__":
    main()
