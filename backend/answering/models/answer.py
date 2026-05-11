from __future__ import annotations

from pydantic import BaseModel, Field


class AnswerCitation(BaseModel):
    index: int
    chunk_id: str
    source_url: str
    title: str
    section_path: list[str] = Field(default_factory=list)


class HallucinationCheck(BaseModel):
    hallucination_risk: float = 0.0
    unsupported_claims: list[str] = Field(default_factory=list)
    safe_to_return: bool = True
    judge_used: bool = False
    judge_error: str | None = None
    judge_input_tokens: int = 0
    judge_output_tokens: int = 0


class AnswerConfidence(BaseModel):
    answer_confidence: float = 0.0
    grounding_confidence: float = 0.0
    hallucination_risk: float = 0.0
    citation_coverage: float = 0.0
    rerank_confidence: float = 0.0


class TokenUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    judge_input_tokens: int = 0
    judge_output_tokens: int = 0
    total_calls: int = 0


class GroundedAnswer(BaseModel):
    query: str
    answer: str
    citations: list[AnswerCitation] = Field(default_factory=list)
    confidence: AnswerConfidence
    grounding_warnings: list[str] = Field(default_factory=list)
    hallucination: HallucinationCheck = Field(default_factory=HallucinationCheck)
    abstained: bool = False
    abstention_reason: str | None = None
    used_chunks: list[str] = Field(default_factory=list)
    rewritten_query: str | None = None
    provider: str = ""
    model: str = ""
    usage: TokenUsage = Field(default_factory=TokenUsage)


ABSTENTION_TEXT = "I could not find reliable information about that in the MITAOE data."
