import logging
import re

import trafilatura
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


class CleanedHtmlResult:
    def __init__(self, html: str, removed_blocks_count: int, multi_article_warning: bool) -> None:
        self.html = html
        self.removed_blocks_count = removed_blocks_count
        self.multi_article_warning = multi_article_warning


class HtmlCleaner:
    BLACKLIST_PATTERNS = [
        "read more",
        "all rights reserved",
        "follow us",
        "click here",
        "view all",
        "contact now",
    ]
    BOILERPLATE_SELECTORS = [
        "nav",
        "footer",
        "script",
        "style",
        "noscript",
        "svg",
        "form",
        "iframe",
        "[role='navigation']",
        "[aria-label*='cookie' i]",
        "[class*='cookie' i]",
        "[id*='cookie' i]",
        "[class*='menu' i]",
        "[id*='menu' i]",
        "[class*='navbar' i]",
        "[id*='navbar' i]",
        "[class*='footer' i]",
        "[id*='footer' i]",
        "[class*='social' i]",
        "[id*='social' i]",
        "[class*='breadcrumb' i]",
        "[id*='breadcrumb' i]",
        "[class*='related' i]",
        "[id*='related' i]",
        "[class*='recent' i]",
        "[id*='recent' i]",
        "[class*='sidebar' i]",
        "[id*='sidebar' i]",
        "[class*='carousel' i]",
        "[id*='carousel' i]",
        "[class*='banner' i]",
        "[id*='banner' i]",
        "[class*='newsletter' i]",
        "[id*='newsletter' i]",
        ".post-navigation",
        ".nav-links",
    ]

    def clean(self, html: str) -> tuple[str, bool]:
        if not html.strip():
            return "", False

        malformed = False
        try:
            result = self.clean_html_result(html)
            cleaned_html = result.html
        except Exception as exc:
            malformed = True
            logger.warning("malformed_html", extra={"extra": {"error": str(exc)}})
            cleaned_html = html

        extracted = trafilatura.extract(
            cleaned_html,
            output_format="markdown",
            include_comments=False,
            include_images=False,
            include_links=False,
            include_tables=True,
            favor_precision=True,
        )
        if extracted:
            fallback = self._soup_to_markdown(cleaned_html)
            chosen = fallback if len(fallback) > len(extracted) * 1.2 else extracted
            return self._clean_text(chosen), malformed

        fallback = self._soup_to_markdown(cleaned_html)
        return self._clean_text(fallback), True

    def clean_html(self, html: str) -> str:
        return self.clean_html_result(html).html

    def clean_html_result(self, html: str) -> CleanedHtmlResult:
        soup = BeautifulSoup(html or "", "html.parser")
        removed = self._remove_boilerplate(soup)
        multi_article_warning = len(soup.find_all(["article", "h1"])) > 2
        primary = self._primary_container(soup)
        return CleanedHtmlResult(str(primary or soup), removed, multi_article_warning)

    def _remove_boilerplate(self, soup: BeautifulSoup) -> int:
        removed = 0
        for selector in self.BOILERPLATE_SELECTORS:
            for tag in soup.select(selector):
                tag.decompose()
                removed += 1
        return removed

    @staticmethod
    def _primary_container(soup: BeautifulSoup):
        candidates = soup.select("article, main, [role='main'], #main, #content, .content, .entry-content, .post-content")
        if not candidates:
            body = soup.body
            return body or soup
        return max(candidates, key=lambda tag: len(tag.get_text(" ", strip=True)))

    def _soup_to_markdown(self, html: str) -> str:
        soup = BeautifulSoup(html, "html.parser")
        lines: list[str] = []

        for element in soup.find_all(["h1", "h2", "h3", "h4", "p", "li", "th", "td"]):
            text = element.get_text(" ", strip=True)
            if not text:
                continue
            if element.name and re.fullmatch(r"h[1-4]", element.name):
                level = int(element.name[1])
                lines.append(f"{'#' * level} {text}")
            elif element.name == "li":
                lines.append(f"- {text}")
            else:
                lines.append(text)

        return "\n".join(lines)

    def _clean_text(self, text: str) -> str:
        text = self._remove_repetitive_lines(text)
        text = re.sub(r"\s+,", ",", text)
        text = re.sub(r",(?=\S)", ", ", text)
        text = re.sub(r"\s+\|+\s*", "\n", text)
        text = re.sub(r"[ \t]+", " ", text)
        return text.strip()

    def _remove_repetitive_lines(self, text: str) -> str:
        kept: list[str] = []
        previous = ""
        for line in text.splitlines():
            line = line.strip()
            normalized = line.lower()
            if not normalized or normalized == previous:
                continue
            if normalized in self.BLACKLIST_PATTERNS:
                continue
            previous = normalized
            kept.append(line)
        return "\n".join(kept)
