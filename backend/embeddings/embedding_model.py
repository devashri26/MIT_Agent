from __future__ import annotations

import numpy as np


DEFAULT_MODEL_NAME = "BAAI/bge-small-en-v1.5"
DEFAULT_DIMENSION = 384


def detect_device() -> str:
    """MPS (Apple Silicon) > CUDA > CPU."""
    import torch

    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class EmbeddingModel:
    """Deterministic sentence-transformers wrapper around BAAI/bge-small-en-v1.5.

    Outputs are L2-normalized so cosine similarity == dot product. The same model is used
    for both passage embedding and query embedding (bge-small-en-v1.5 is a unified model).
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        device: str | None = None,
    ) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.device = device or detect_device()
        self._model = SentenceTransformer(model_name, device=self.device)
        get_dim = getattr(self._model, "get_embedding_dimension", None) or self._model.get_sentence_embedding_dimension
        self.dimension = int(get_dim())

    def embed(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        vectors = self._model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return vectors.astype(np.float32, copy=False)

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed([text])[0]
