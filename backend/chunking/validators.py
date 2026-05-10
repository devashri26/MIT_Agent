import regex as re

from backend.chunking.models.chunk import SemanticChunk


class ChunkValidator:
    BOILERPLATE_PATTERNS = [
        r"^\s*(apply now|read more|follow us|click here)\s*$",
        r"all rights reserved",
        r"\bniaa\b.*\bhey aspirant\b",
        r"^@?\s*\d{4}\s+mit academy of engineering$",
        r"404-error-page",
    ]

    def validate(self, chunk: SemanticChunk, hard_max_tokens: int = 1_000) -> list[str]:
        return self.validate_text(chunk.text, chunk.token_count, hard_max_tokens)

    def validate_text(self, text: str, token_count: int, hard_max_tokens: int = 1_000) -> list[str]:
        warnings: list[str] = []
        if not text.strip():
            warnings.append("empty_chunk")
        if token_count > hard_max_tokens:
            warnings.append("token_overflow")
        if self.is_boilerplate_only(text):
            warnings.append("boilerplate_heavy")
        return warnings

    def should_reject(self, text: str) -> bool:
        return not text.strip() or self.is_boilerplate_only(text) or self._is_useless_fragment(text)

    def is_boilerplate_only(self, text: str) -> bool:
        normalized = " ".join(text.lower().split())
        if len(normalized) > 160:
            return False
        return any(re.search(pattern, normalized, flags=re.IGNORECASE) for pattern in self.BOILERPLATE_PATTERNS)

    @staticmethod
    def _is_useless_fragment(text: str) -> bool:
        normalized = " ".join(text.strip().split())
        if re.fullmatch(r"[\d\W]+", normalized):
            return True
        if len(normalized) < 20 and re.fullmatch(r"(the|a|an|and|or|of|in|on|to|for|with|field)\b.*", normalized, re.IGNORECASE):
            return True
        return False
