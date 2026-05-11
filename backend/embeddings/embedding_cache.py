from __future__ import annotations

from pathlib import Path

import numpy as np


class EmbeddingCache:
    """Disk-backed cache keyed by chunk_hash. Stored as a single NPZ with parallel arrays
    `hashes` (variable-length unicode) and `vectors` (N x D float32). Loaded lazily on first
    lookup; persisted on save()."""

    def __init__(self, cache_path: Path = Path("datasets/embedding_cache.npz")) -> None:
        self.cache_path = cache_path
        self._hash_to_vector: dict[str, np.ndarray] = {}
        self._dimension: int | None = None
        self._loaded = False

    def _load(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        if not self.cache_path.exists():
            return
        with np.load(self.cache_path, allow_pickle=False) as archive:
            hashes = archive["hashes"]
            vectors = archive["vectors"]
            for h, v in zip(hashes.tolist(), vectors):
                self._hash_to_vector[str(h)] = v.astype(np.float32, copy=False)
            if vectors.shape[0] > 0:
                self._dimension = int(vectors.shape[1])

    def get(self, chunk_hash: str) -> np.ndarray | None:
        self._load()
        return self._hash_to_vector.get(chunk_hash)

    def put(self, chunk_hash: str, vector: np.ndarray) -> None:
        self._load()
        vec = vector.astype(np.float32, copy=False)
        if self._dimension is None:
            self._dimension = int(vec.shape[0])
        elif vec.shape[0] != self._dimension:
            raise ValueError(
                f"vector dimension {vec.shape[0]} does not match cache dimension {self._dimension}"
            )
        self._hash_to_vector[chunk_hash] = vec

    def __contains__(self, chunk_hash: str) -> bool:
        self._load()
        return chunk_hash in self._hash_to_vector

    def __len__(self) -> int:
        self._load()
        return len(self._hash_to_vector)

    def save(self) -> None:
        self._load()
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        if not self._hash_to_vector:
            return
        hashes = np.array(list(self._hash_to_vector.keys()))
        vectors = np.stack(list(self._hash_to_vector.values()))
        tmp_path = self.cache_path.with_suffix(self.cache_path.suffix + ".tmp")
        with tmp_path.open("wb") as out_file:
            np.savez(out_file, hashes=hashes, vectors=vectors)
        tmp_path.replace(self.cache_path)
