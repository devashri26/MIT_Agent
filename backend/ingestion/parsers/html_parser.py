import re

from bs4 import BeautifulSoup

from backend.ingestion.models.document import DocumentSection


class HtmlParser:
    def headings(self, html: str) -> list[str]:
        soup = BeautifulSoup(html or "", "html.parser")
        values: list[str] = []
        for tag in soup.find_all(re.compile(r"^h[1-6]$")):
            text = tag.get_text(" ", strip=True)
            if text:
                values.append(text)
        return values

    def sections(self, html: str, fallback_text: str) -> list[DocumentSection]:
        soup = BeautifulSoup(html or "", "html.parser")
        sections: list[DocumentSection] = []
        current_heading = "Overview"
        current_lines: list[str] = []

        for element in soup.find_all(["h1", "h2", "h3", "p", "li", "th", "td"]):
            text = element.get_text(" ", strip=True)
            if not text:
                continue
            if element.name in {"h1", "h2", "h3"}:
                self._append_section(sections, current_heading, current_lines)
                current_heading = text
                current_lines = []
            elif element.name == "li":
                current_lines.append(f"- {text}")
            else:
                current_lines.append(text)

        self._append_section(sections, current_heading, current_lines)
        if sections:
            return self._split_large_sections(sections)
        return self._fallback_sections(fallback_text)

    @staticmethod
    def _append_section(sections: list[DocumentSection], heading: str, lines: list[str]) -> None:
        content = "\n".join(line for line in lines if line.strip()).strip()
        if content:
            sections.append(DocumentSection(heading=heading.strip()[:140], content=content))

    @staticmethod
    def _fallback_sections(text: str) -> list[DocumentSection]:
        paragraphs = [part.strip() for part in re.split(r"\n{2,}", text) if part.strip()]
        if not paragraphs:
            return []
        sections: list[DocumentSection] = []
        heading = "Overview"
        current: list[str] = []
        heading_pattern = re.compile(r"^(\d+[\.)]\s+.+|[A-Z][A-Za-z &/-]{3,80}:?)$")
        for paragraph in paragraphs:
            first_line = paragraph.splitlines()[0].strip()
            if heading_pattern.match(first_line) and current:
                HtmlParser._append_section(sections, heading, current)
                heading = first_line.strip(":")
                current = paragraph.splitlines()[1:] or []
            else:
                current.append(paragraph)
        HtmlParser._append_section(sections, heading, current)
        return HtmlParser._split_large_sections(sections)

    @staticmethod
    def _split_large_sections(sections: list[DocumentSection], max_words: int = 1_200) -> list[DocumentSection]:
        split_sections: list[DocumentSection] = []
        for section in sections:
            words = section.content.split()
            if len(words) <= max_words:
                split_sections.append(section)
                continue
            chunk: list[str] = []
            part = 1
            for word in words:
                chunk.append(word)
                if len(chunk) >= max_words:
                    split_sections.append(
                        DocumentSection(heading=f"{section.heading} Part {part}", content=" ".join(chunk))
                    )
                    part += 1
                    chunk = []
            if chunk:
                split_sections.append(DocumentSection(heading=f"{section.heading} Part {part}", content=" ".join(chunk)))
        return split_sections
