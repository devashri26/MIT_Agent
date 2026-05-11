from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import regex as re


TRACKING_PARAMS = {
    "utm_source",
    "utm_medium",
    "utm_campaign",
    "utm_term",
    "utm_content",
    "fbclid",
    "gclid",
    "msclkid",
    "ref",
    "referrer",
}

SORT_PARAMS = {"c", "o", "sort", "order", "page"}


def canonicalize_url(url: str) -> str:
    """Deterministic URL canonicalization: lowercase host, strip www, drop trackers, sort query."""
    if not url:
        return ""
    parsed = urlparse(url)
    scheme = (parsed.scheme or "https").lower()
    host = (parsed.netloc or "").lower()
    if host.startswith("www."):
        host = host[4:]

    path = re.sub(r"/+", "/", parsed.path or "/")
    if len(path) > 1 and path.endswith("/"):
        path = path.rstrip("/")

    params = [
        (k, v)
        for k, v in parse_qsl(parsed.query, keep_blank_values=False)
        if k.lower() not in TRACKING_PARAMS and k.lower() not in SORT_PARAMS
    ]
    params.sort()
    query = urlencode(params)

    return urlunparse((scheme, host, path, "", query, ""))


class CanonicalIndex:
    """Tracks first-seen canonical URLs and chunk hashes across the corpus.

    A canonical URL is claimed by the first original URL that maps to it; subsequent
    original URLs that map to the same canonical are flagged non_canonical. Re-registering
    the same original URL (multiple chunks from one document) returns its existing status.
    """

    def __init__(self) -> None:
        self._canonical_winner: dict[str, str] = {}
        self._seen_chunk_hashes: set[str] = set()
        self.url_to_canonical: dict[str, str] = {}

    def register_url(self, url: str) -> tuple[str, bool]:
        canonical = canonicalize_url(url)
        self.url_to_canonical[url] = canonical
        winner = self._canonical_winner.get(canonical)
        if winner is None:
            self._canonical_winner[canonical] = url
            return canonical, True
        return canonical, winner == url

    def register_chunk_hash(self, chunk_hash: str) -> bool:
        if not chunk_hash:
            return True
        if chunk_hash in self._seen_chunk_hashes:
            return False
        self._seen_chunk_hashes.add(chunk_hash)
        return True
