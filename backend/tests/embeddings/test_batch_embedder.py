from pathlib import Path

import numpy as np

from backend.embeddings.batch_embedder import BatchEmbedder
from backend.embeddings.embedding_cache import EmbeddingCache


def test_batch_embedder_uses_cache(tmp_path: Path, fake_model) -> None:
    cache = EmbeddingCache(cache_path=tmp_path / "cache.npz")
    embedder = BatchEmbedder(model=fake_model, cache=cache)

    first = embedder.embed(["h1", "h2"], ["foo", "bar"])
    assert first.shape == (2, fake_model.dimension)
    assert len(cache) == 2

    second = embedder.embed(["h1", "h2"], ["foo-different-text", "bar-different-text"])
    # Cached vectors win — text isn't re-embedded
    assert np.allclose(first, second)


def test_batch_embedder_mixed_hits_and_misses(tmp_path: Path, fake_model) -> None:
    cache = EmbeddingCache(cache_path=tmp_path / "cache.npz")
    cache.put("hit", fake_model.embed(["cached"])[0])

    embedder = BatchEmbedder(model=fake_model, cache=cache)
    vectors = embedder.embed(["hit", "miss"], ["cached", "miss-text"])
    assert vectors.shape == (2, fake_model.dimension)
    # Both should match what the fake_model would produce now
    expected_miss = fake_model.embed(["miss-text"])[0]
    assert np.allclose(vectors[1], expected_miss)
    assert "miss" in cache


def test_batch_embedder_no_cache(fake_model) -> None:
    embedder = BatchEmbedder(model=fake_model, cache=None)
    vectors = embedder.embed(["h1", "h2"], ["a", "b"])
    assert vectors.shape == (2, fake_model.dimension)


def test_batch_embedder_empty_input(fake_model) -> None:
    embedder = BatchEmbedder(model=fake_model, cache=None)
    vectors = embedder.embed([], [])
    assert vectors.shape == (0, fake_model.dimension)
