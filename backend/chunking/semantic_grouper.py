import regex as re

from backend.chunking.token_splitter import TokenSplitter


class SemanticGrouper:
    def __init__(self, token_splitter: TokenSplitter, target_tokens: int = 650) -> None:
        self.token_splitter = token_splitter
        self.target_tokens = target_tokens

    def group(self, heading: str, text: str, content_type: str) -> list[str]:
        if content_type == "FAQ":
            return self._faq_groups(text)
        blocks = [block.strip() for block in re.split(r"\n{2,}", text) if block.strip()]
        if not blocks:
            return []
        groups: list[str] = []
        current: list[str] = []
        for block in blocks:
            proposed = "\n\n".join([*current, block])
            if current and self.token_splitter.count_tokens(proposed) > self.target_tokens:
                groups.append("\n\n".join(current).strip())
                current = [block]
            else:
                current.append(block)
        if current:
            groups.append("\n\n".join(current).strip())
        return groups

    def _faq_groups(self, text: str) -> list[str]:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        groups: list[str] = []
        current: list[str] = []
        question_pattern = re.compile(r"^(\d+[\.)]\s*)?(q\.|question|\w.+\?)", re.IGNORECASE)
        for line in lines:
            if current and question_pattern.match(line):
                groups.append("\n".join(current).strip())
                current = [line]
            else:
                current.append(line)
        if current:
            groups.append("\n".join(current).strip())
        return groups

