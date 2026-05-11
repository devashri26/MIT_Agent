from __future__ import annotations

import numpy as np

from backend.embeddings.embedding_cache import EmbeddingCache
from backend.embeddings.embedding_model import EmbeddingModel


class BatchEmbedder:
    """Cache-aware batched embedding. Returns vectors in the input order. Texts already
    present in the cache (by their hash key) skip the model entirely; misses are batched
    through the model and back-filled."""

    def __init__(
        self,
        model: EmbeddingModel,
        cache: EmbeddingCache | None = None,
        batch_size: int = 32,
    ) -> None:
        self.model = model
        self.cache = cache
        self.batch_size = batch_size

    def embed(self, hashes: list[str], texts: list[str]) -> np.ndarray:
        if len(hashes) != len(texts):
            raise ValueError("hashes and texts must have the same length")
        if not hashes:
            return np.zeros((0, self.model.dimension), dtype=np.float32)

        output = np.zeros((len(hashes), self.model.dimension), dtype=np.float32)
        miss_indices: list[int] = []
        miss_texts: list[str] = []

        for idx, (h, text) in enumerate(zip(hashes, texts)):
            cached = self.cache.get(h) if self.cache is not None else None
            if cached is not None:
                output[idx] = cached
            else:
                miss_indices.append(idx)
                miss_texts.append(text)

        if miss_texts:
            new_vectors = self.model.embed(miss_texts, batch_size=self.batch_size)
            for slot, idx in enumerate(miss_indices):
                vector = new_vectors[slot]
                output[idx] = vector
                if self.cache is not None:
                    self.cache.put(hashes[idx], vector)

        return output

    @property
    def cache_hits(self) -> int:
        return len(self.cache) if self.cache is not None else 0
