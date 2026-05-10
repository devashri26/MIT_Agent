import hashlib
from urllib.parse import urldefrag

from rapidfuzz import fuzz


class Deduplicator:
    def __init__(self) -> None:
        self._seen_urls: dict[str, str] = {}
        self._seen_content_hashes: dict[str, str] = {}
        self._fingerprints: list[tuple[str, str]] = []

    def is_duplicate(self, url: str, content: str) -> tuple[bool, str | None, str | None]:
        canonical_url = self._canonical_url(url)
        if canonical_url in self._seen_urls:
            return True, "url", self._seen_urls[canonical_url]

        content_hash = self.content_hash(content)
        if content_hash in self._seen_content_hashes:
            return True, "content_hash", self._seen_content_hashes[content_hash]

        self._seen_urls[canonical_url] = url
        self._seen_content_hashes[content_hash] = url
        self._fingerprints.append((self._fingerprint(content), url))
        return False, None, None

    def near_duplicate_match(self, url: str, content: str, threshold: float = 95.0) -> str | None:
        fingerprint = self._fingerprint(content)
        for seen_fingerprint, seen_url in self._fingerprints:
            if seen_url == url:
                continue
            if fuzz.token_set_ratio(fingerprint, seen_fingerprint) > threshold:
                return seen_url
        return None

    @staticmethod
    def content_hash(content: str) -> str:
        normalized = " ".join(content.lower().split())
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    @staticmethod
    def page_id(url: str, content: str) -> str:
        seed = f"{Deduplicator._canonical_url(url)}:{Deduplicator.content_hash(content)}"
        return hashlib.sha256(seed.encode("utf-8")).hexdigest()[:24]

    @staticmethod
    def _canonical_url(url: str) -> str:
        clean_url, _ = urldefrag(url.strip())
        return clean_url.rstrip("/").lower()

    @staticmethod
    def _fingerprint(content: str) -> str:
        words = " ".join(content.lower().split()).split(" ")
        return " ".join(words[:250])
