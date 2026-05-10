from typing import Any


class SectionSplitter:
    def split(self, document: dict[str, Any]) -> list[dict[str, str]]:
        sections = document.get("sections") or []
        usable = [
            {"heading": str(section.get("heading") or "Overview"), "content": str(section.get("content") or "").strip()}
            for section in sections
            if str(section.get("content") or "").strip()
        ]
        if usable:
            return usable
        clean_content = str(document.get("clean_content") or "").strip()
        return [{"heading": "Overview", "content": clean_content}] if clean_content else []

