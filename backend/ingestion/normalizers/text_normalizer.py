import re
import unicodedata


class TextNormalizer:
    BLACKLIST_PATTERNS = [
        r"\bread more\b",
        r"\ball rights reserved\b",
        r"\bfollow us\b",
        r"\bclick here\b",
    ]

    def normalize(self, text: str) -> str:
        normalized = unicodedata.normalize("NFKC", text)
        normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
        normalized = normalized.replace("\u00a0", " ")
        normalized = normalized.encode("utf-8", errors="ignore").decode("utf-8")
        normalized = re.sub(r"â€™", "'", normalized)
        normalized = re.sub(r"â€“|â€”", "-", normalized)
        normalized = re.sub(r"â€œ|â€", '"', normalized)
        normalized = re.sub(r"\s+,", ",", normalized)
        normalized = re.sub(r",(?=\S)", ", ", normalized)
        normalized = re.sub(r"\s*\|\s*", "\n", normalized)
        normalized = re.sub(r"[ \t\f\v]+", " ", normalized)
        normalized = re.sub(r" *\n *", "\n", normalized)
        normalized = re.sub(r"\n{3,}", "\n\n", normalized)
        return self._remove_bad_lines(normalized).strip()

    def _remove_bad_lines(self, text: str) -> str:
        kept: list[str] = []
        previous = ""
        blank_kept = False
        for line in text.splitlines():
            cleaned = line.strip()
            lowered = cleaned.lower()
            if not cleaned:
                if kept and not blank_kept:
                    kept.append("")
                    blank_kept = True
                continue
            blank_kept = False
            if lowered == previous:
                continue
            if any(re.search(pattern, lowered) for pattern in self.BLACKLIST_PATTERNS):
                continue
            previous = lowered
            kept.append(cleaned)
        return "\n".join(kept)
