import regex as re
import spacy
import tiktoken


class TokenSplitter:
    def __init__(self, ideal_min: int = 400, ideal_max: int = 700, hard_max: int = 1_000, overlap: int = 60) -> None:
        self.ideal_min = ideal_min
        self.ideal_max = ideal_max
        self.hard_max = hard_max
        self.overlap = overlap
        self.encoding = tiktoken.get_encoding("cl100k_base")
        try:
            self.nlp = spacy.load("en_core_web_sm", disable=["ner", "tagger", "lemmatizer"])
        except OSError:
            self.nlp = spacy.blank("en")
            self.nlp.add_pipe("sentencizer")

    def count_tokens(self, text: str) -> int:
        return len(self.encoding.encode(text))

    def split_text(self, text: str) -> list[str]:
        if self.count_tokens(text) <= self.hard_max:
            return [text.strip()] if text.strip() else []

        units = self._semantic_units(text)
        chunks: list[str] = []
        current: list[str] = []
        current_tokens = 0

        for unit in units:
            unit_tokens = self.count_tokens(unit)
            if unit_tokens > self.hard_max:
                chunks.extend(self._split_large_unit(unit))
                continue
            if current and current_tokens + unit_tokens > self.ideal_max:
                chunks.append("\n\n".join(current).strip())
                current = self._overlap_units(current)
                current_tokens = self.count_tokens("\n\n".join(current)) if current else 0
            current.append(unit)
            current_tokens += unit_tokens

        if current:
            chunks.append("\n\n".join(current).strip())
        return [chunk for chunk in chunks if chunk]

    def _semantic_units(self, text: str) -> list[str]:
        blocks = [block.strip() for block in re.split(r"\n{2,}", text) if block.strip()]
        units: list[str] = []
        for block in blocks:
            if self._is_atomic_block(block):
                units.append(block)
            else:
                doc = self.nlp(block)
                sentences = [sentence.text.strip() for sentence in doc.sents if sentence.text.strip()]
                units.extend(sentences or [block])
        return units

    def _split_large_unit(self, text: str) -> list[str]:
        doc = self.nlp(text)
        sentences = [sentence.text.strip() for sentence in doc.sents if sentence.text.strip()]
        if not sentences:
            words = text.split()
            return [" ".join(words[index : index + 700]) for index in range(0, len(words), 700)]
        chunks: list[str] = []
        current: list[str] = []
        for sentence in sentences:
            proposed = " ".join([*current, sentence]).strip()
            if current and self.count_tokens(proposed) > self.hard_max:
                chunks.append(" ".join(current).strip())
                current = self._overlap_sentences(current)
            current.append(sentence)
        if current:
            chunks.append(" ".join(current).strip())
        return chunks

    def _overlap_units(self, units: list[str]) -> list[str]:
        overlap: list[str] = []
        for unit in reversed(units):
            proposed = [unit, *overlap]
            if self.count_tokens("\n\n".join(proposed)) > self.overlap:
                break
            overlap = proposed
        return overlap

    def _overlap_sentences(self, sentences: list[str]) -> list[str]:
        overlap: list[str] = []
        for sentence in reversed(sentences):
            proposed = [sentence, *overlap]
            if self.count_tokens(" ".join(proposed)) > self.overlap:
                break
            overlap = proposed
        return overlap

    @staticmethod
    def _is_atomic_block(text: str) -> bool:
        lowered = text.lower()
        bullet_lines = sum(1 for line in text.splitlines() if line.strip().startswith(("-", "*", "•")))
        return (
            "|" in text
            or "\t" in text
            or bullet_lines >= 2
            or bool(re.search(r"\bsemester\s+[ivx0-9]+", lowered))
            or bool(re.search(r"^(q\.|question|answer|ans\.)", lowered))
        )

