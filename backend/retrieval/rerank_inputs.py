from dataclasses import asdict

from backend.retrieval.fusion import FusedItem


def fused_to_rerank_input(fused: list[FusedItem]) -> list[dict]:
    """Project FusedItem records into a plain-dict shape suitable for a future reranker
    layer (cross-encoder or LLM-based). No reranker is wired up yet — this exists so the
    hand-off contract is established before that phase."""
    return [asdict(item) for item in fused]
