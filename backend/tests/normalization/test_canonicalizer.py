from backend.normalization.canonicalizer import CanonicalIndex, canonicalize_url


def test_canonicalize_lowercases_host() -> None:
    assert canonicalize_url("HTTPS://Example.EDU/Path") == "https://example.edu/Path"


def test_canonicalize_strips_www() -> None:
    assert canonicalize_url("https://www.example.edu/page") == "https://example.edu/page"


def test_canonicalize_strips_trailing_slash() -> None:
    assert canonicalize_url("https://example.edu/page/") == "https://example.edu/page"


def test_canonicalize_drops_tracking_params() -> None:
    canonical = canonicalize_url("https://example.edu/page?utm_source=x&id=42")
    assert "utm_source" not in canonical
    assert "id=42" in canonical


def test_canonicalize_drops_sort_params() -> None:
    canonical = canonicalize_url("https://example.edu/files/?c=n&o=a")
    assert "c=" not in canonical
    assert "o=" not in canonical


def test_canonical_index_first_url_canonical() -> None:
    index = CanonicalIndex()
    _, first = index.register_url("https://example.edu/page")
    _, second = index.register_url("https://www.example.edu/page/")
    assert first is True
    assert second is False


def test_canonical_index_same_url_twice_stays_canonical() -> None:
    """Multiple chunks from one document re-register the same URL — must stay canonical."""
    index = CanonicalIndex()
    _, first = index.register_url("https://example.edu/page")
    _, second = index.register_url("https://example.edu/page")
    assert first is True
    assert second is True


def test_canonical_index_chunk_hash_dedup() -> None:
    index = CanonicalIndex()
    assert index.register_chunk_hash("abc") is True
    assert index.register_chunk_hash("abc") is False
    assert index.register_chunk_hash("def") is True
