from __future__ import annotations

from typing import Sequence

import numpy as np


DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-base"


def detect_device() -> str:
    """MPS > CUDA > CPU. Mirrors the embedding model's policy."""
    import torch

    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class RerankerModel:
    """sentence-transformers CrossEncoder wrapper around BAAI/bge-reranker-base.

    Returns raw logits (unbounded); caller applies sigmoid via score_calibrator for the
    [0,1] calibrated score. Singleton-friendly: hold one instance per process.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_RERANKER_MODEL,
        device: str | None = None,
        max_length: int = 512,
    ) -> None:
        from sentence_transformers import CrossEncoder

        self.model_name = model_name
        self.device = device or detect_device()
        self.max_length = max_length
        self._model = CrossEncoder(model_name, device=self.device, max_length=max_length)

    def score(self, query: str, passages: Sequence[str], batch_size: int = 16) -> np.ndarray:
        if not passages:
            return np.zeros((0,), dtype=np.float32)
        pairs = [[query, passage] for passage in passages]
        scores = self._model.predict(
            pairs,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return np.asarray(scores, dtype=np.float32).reshape(-1)
