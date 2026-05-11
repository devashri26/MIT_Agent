from pathlib import Path

import numpy as np

from backend.embeddings.embedding_cache import EmbeddingCache


def test_cache_roundtrip(tmp_path: Path) -> None:
    cache_path = tmp_path / "cache.npz"
    cache = EmbeddingCache(cache_path=cache_path)

    v1 = np.ones(384, dtype=np.float32) * 0.5
    v2 = np.ones(384, dtype=np.float32) * 0.25
    cache.put("h1", v1)
    cache.put("h2", v2)
    cache.save()

    reloaded = EmbeddingCache(cache_path=cache_path)
    assert "h1" in reloaded
    assert "h2" in reloaded
    assert len(reloaded) == 2
    assert np.allclose(reloaded.get("h1"), v1)
    assert np.allclose(reloaded.get("h2"), v2)


def test_cache_missing_returns_none(tmp_path: Path) -> None:
    cache = EmbeddingCache(cache_path=tmp_path / "nope.npz")
    assert cache.get("missing") is None
    assert "missing" not in cache
    assert len(cache) == 0


def test_cache_rejects_dimension_mismatch(tmp_path: Path) -> None:
    cache = EmbeddingCache(cache_path=tmp_path / "cache.npz")
    cache.put("h1", np.zeros(384, dtype=np.float32))
    try:
        cache.put("h2", np.zeros(128, dtype=np.float32))
    except ValueError as exc:
        assert "dimension" in str(exc).lower()
    else:
        raise AssertionError("expected ValueError for dimension mismatch")


def test_cache_save_skips_when_empty(tmp_path: Path) -> None:
    cache_path = tmp_path / "cache.npz"
    EmbeddingCache(cache_path=cache_path).save()
    assert not cache_path.exists()
