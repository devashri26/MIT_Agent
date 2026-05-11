import hashlib

import numpy as np
import pytest


class FakeEmbeddingModel:
    """Deterministic stand-in for EmbeddingModel. Hashes each text to seed an L2-normalized
    random vector. Used by unit tests so we don't load BAAI/bge-small in CI."""

    model_name = "fake/test-embedder"
    device = "cpu"
    dimension = 384

    def embed(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        vectors = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for i, text in enumerate(texts):
            seed = int(hashlib.md5(text.encode("utf-8")).hexdigest()[:8], 16)
            rng = np.random.default_rng(seed)
            vector = rng.standard_normal(self.dimension).astype(np.float32)
            norm = np.linalg.norm(vector)
            if norm > 0:
                vector = vector / norm
            vectors[i] = vector
        return vectors

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed([text])[0]


@pytest.fixture
def fake_model() -> FakeEmbeddingModel:
    return FakeEmbeddingModel()


class FakeRerankerModel:
    """Deterministic stand-in for RerankerModel. Scores higher when the query terms appear
    in the passage (Jaccard-ish over lowercased word sets). Used so unit tests don't load
    BAAI/bge-reranker-base."""

    model_name = "fake/test-reranker"
    device = "cpu"

    def score(self, query: str, passages: list[str], batch_size: int = 16) -> np.ndarray:
        query_tokens = set(query.lower().split())
        scores = np.zeros(len(passages), dtype=np.float32)
        for i, passage in enumerate(passages):
            passage_tokens = set(passage.lower().split())
            if not query_tokens or not passage_tokens:
                scores[i] = -5.0
                continue
            overlap = len(query_tokens & passage_tokens)
            jaccard = overlap / len(query_tokens | passage_tokens)
            scores[i] = float(-3.0 + 8.0 * jaccard)
        return scores


@pytest.fixture
def fake_reranker() -> FakeRerankerModel:
    return FakeRerankerModel()
