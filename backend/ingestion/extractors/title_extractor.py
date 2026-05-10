from dataclasses import dataclass
from urllib.parse import unquote, urlparse

import regex as re
from bs4 import BeautifulSoup
from rapidfuzz import fuzz

from backend.ingestion.models.document import RawWebsiteRow, ValidationIssue


@dataclass(frozen=True)
class TitleExtractionResult:
    title: str
    source: str
    issues: list[ValidationIssue]
    similarity: float | None


class TitleExtractor:
    MIN_TITLE_LENGTH = 6
    URL_TITLE_WARNING_THRESHOLD = 35.0

    def extract(self, row: RawWebsiteRow) -> TitleExtractionResult:
        candidates = [
            ("og:title", row.metadata.get("og_title", "")),
            ("article_h1", self._article_h1(row.html)),
            ("main_h1", self._main_h1(row.html)),
            ("html_title", self._html_title(row.html) or row.metadata.get("metadata_title", "")),
            ("url_slug", self._title_from_url(row.url)),
        ]

        title = ""
        source = "missing"
        for candidate_source, candidate in candidates:
            cleaned = self._clean_title(candidate)
            if cleaned:
                title = cleaned
                source = candidate_source
                break

        issues: list[ValidationIssue] = []
        if len(title) < self.MIN_TITLE_LENGTH:
            issues.append(ValidationIssue(field="title", message="title is too short"))

        similarity = self.url_title_similarity(row.url, title) if title else None
        if similarity is not None and similarity < self.URL_TITLE_WARNING_THRESHOLD:
            issues.append(
                ValidationIssue(
                    field="title",
                    message="title has low similarity to URL slug",
                    severity="warning",
                )
            )

        return TitleExtractionResult(title=title, source=source, issues=issues, similarity=similarity)

    @staticmethod
    def url_title_similarity(url: str, title: str) -> float | None:
        slug = TitleExtractor._url_slug(url)
        if not slug or not title:
            return None
        return float(fuzz.token_set_ratio(slug, title))

    @staticmethod
    def _article_h1(html: str) -> str:
        soup = BeautifulSoup(html or "", "html.parser")
        container = soup.find("article") or soup.select_one("main, [role='main'], .content, #content")
        h1 = container.find("h1") if container else None
        return h1.get_text(" ", strip=True) if h1 else ""

    @staticmethod
    def _main_h1(html: str) -> str:
        soup = BeautifulSoup(html or "", "html.parser")
        h1 = soup.select_one("main h1, [role='main'] h1") or soup.find("h1")
        return h1.get_text(" ", strip=True) if h1 else ""

    @staticmethod
    def _html_title(html: str) -> str:
        soup = BeautifulSoup(html or "", "html.parser")
        title = soup.find("title")
        return title.get_text(" ", strip=True) if title else ""

    @staticmethod
    def _clean_title(title: str) -> str:
        title = re.sub(r"\s+", " ", title or "").strip(" -|")
        title = re.sub(r"\s+\|\s+MITAOE?$", "", title, flags=re.IGNORECASE)
        return title[:180]

    @staticmethod
    def _url_slug(url: str) -> str:
        path = unquote(urlparse(url).path)
        name = path.rstrip("/").split("/")[-1]
        name = re.sub(r"\.(php|html?|aspx?)$", "", name, flags=re.IGNORECASE)
        name = re.sub(r"[-_]+", " ", name)
        return name.strip()

    @staticmethod
    def _title_from_url(url: str) -> str:
        slug = TitleExtractor._url_slug(url)
        return " ".join(word.capitalize() for word in slug.split()) if slug else "Untitled"
