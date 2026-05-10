from pathlib import Path
from typing import Any

import orjson
import regex as re
from rank_bm25 import BM25Okapi

from backend.retrieval.models.search import RetrievedChunk, SearchResponse


class BM25RetrievalService:
    def __init__(self, chunks_path: Path = Path("datasets/chunks.jsonl")) -> None:
        self.chunks_path = chunks_path
        self.chunks = self._load_chunks(chunks_path)
        self.tokenized_corpus = [self._tokenize(self._search_text(chunk)) for chunk in self.chunks]
        self.index = BM25Okapi(self.tokenized_corpus) if self.tokenized_corpus else None

    def search(self, query: str, top_k: int = 5) -> SearchResponse:
        query_tokens = self._tokenize(query)
        if not query_tokens or self.index is None:
            return SearchResponse(query=query, top_k=top_k, results=[])

        scores = self.index.get_scores(query_tokens)
        ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)[:top_k]
        results: list[RetrievedChunk] = []
        for rank, (index, score) in enumerate(ranked, start=1):
            chunk = self.chunks[index]
            results.append(
                RetrievedChunk(
                    rank=rank,
                    score=round(float(score), 4),
                    chunk_id=chunk["chunk_id"],
                    document_id=chunk["document_id"],
                    url=chunk["url"],
                    title=chunk["title"],
                    department=chunk.get("department"),
                    page_type=chunk["page_type"],
                    section_heading=chunk["section_heading"],
                    token_count=chunk["token_count"],
                    content_type=chunk["content_type"],
                    quality_score=chunk["quality_score"],
                    text=chunk["text"],
                    metadata=chunk.get("metadata") or {},
                )
            )
        return SearchResponse(query=query, top_k=top_k, results=results)

    @staticmethod
    def _load_chunks(path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        return [orjson.loads(line) for line in path.read_bytes().splitlines() if line.strip()]

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"[\p{L}\p{N}][\p{L}\p{N}&.+-]*", text.lower())

    @staticmethod
    def _search_text(chunk: dict[str, Any]) -> str:
        metadata = chunk.get("metadata") or {}
        headings = " ".join(metadata.get("headings") or [])
        return " ".join(
            [
                chunk.get("title") or "",
                chunk.get("section_heading") or "",
                headings,
                chunk.get("page_type") or "",
                chunk.get("content_type") or "",
                chunk.get("text") or "",
            ]
        )
