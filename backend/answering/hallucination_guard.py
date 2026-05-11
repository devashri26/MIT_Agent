from __future__ import annotations

import re

import orjson

from backend.answering.models.answer import HallucinationCheck
from backend.context.validators import GroundedContext
from backend.llm.prompts.grounded_answering import HALLUCINATION_JUDGE_PROMPT
from backend.llm.provider_interface import BaseLLMProvider
from backend.llm.validators import LLMMessage, LLMRequest


_JSON_OBJECT_PATTERN = re.compile(r"\{.*\}", re.DOTALL)


def _serialize_context(grounded_context: GroundedContext) -> str:
    parts: list[str] = []
    for index, block in enumerate(grounded_context.context_blocks, start=1):
        parts.append(f"[{index}] {block.text.strip()}")
    return "\n\n".join(parts)


def _parse_judge_output(text: str) -> tuple[float, list[str], str | None]:
    """Tolerant JSON parse. Tries direct parse first, then a regex-sliced object as a
    fallback. Returns (risk, unsupported_claims, parse_error)."""
    if not text:
        return 0.0, [], "empty_response"
    try:
        data = orjson.loads(text)
    except orjson.JSONDecodeError:
        match = _JSON_OBJECT_PATTERN.search(text)
        if not match:
            return 0.0, [], "no_json_object"
        try:
            data = orjson.loads(match.group(0))
        except orjson.JSONDecodeError as exc:
            return 0.0, [], f"json_decode_error:{exc}"
    if not isinstance(data, dict):
        return 0.0, [], "non_object_response"
    risk = data.get("hallucination_risk", 0.0)
    try:
        risk = float(risk)
    except (TypeError, ValueError):
        risk = 0.0
    risk = max(0.0, min(1.0, risk))
    claims = data.get("unsupported_claims", []) or []
    claims = [str(c) for c in claims if isinstance(c, (str, int, float))]
    return risk, claims, None


def validate_grounded_answer(
    provider: BaseLLMProvider,
    answer: str,
    grounded_context: GroundedContext,
    judge_model: str = "",
    max_risk_for_safe: float = 0.5,
) -> HallucinationCheck:
    """LLM-as-judge check. Same provider as the answerer by default, so config stays
    simple. On parse failure we surface judge_error but do not block the answer (better
    to ship with a warning than swallow legitimate output)."""
    if not answer.strip():
        return HallucinationCheck(safe_to_return=False, judge_used=False, judge_error="empty_answer")

    context_text = _serialize_context(grounded_context)
    if not context_text:
        return HallucinationCheck(
            hallucination_risk=1.0,
            safe_to_return=False,
            judge_used=False,
            judge_error="no_context_to_judge_against",
        )

    request = LLMRequest(
        system_prompt=HALLUCINATION_JUDGE_PROMPT,
        messages=[
            LLMMessage(
                role="user",
                content=f"ANSWER:\n{answer}\n\nCONTEXT:\n{context_text}",
            )
        ],
        model=judge_model or provider.default_model,
        temperature=0.0,
        max_tokens=400,
        response_format="json",
    )
    try:
        response = provider.generate(request)
    except Exception as exc:
        return HallucinationCheck(
            hallucination_risk=0.0,
            safe_to_return=True,
            judge_used=False,
            judge_error=f"provider_error:{exc}",
        )

    risk, claims, parse_error = _parse_judge_output(response.text)
    return HallucinationCheck(
        hallucination_risk=round(risk, 4),
        unsupported_claims=claims,
        safe_to_return=risk < max_risk_for_safe,
        judge_used=True,
        judge_error=parse_error,
        judge_input_tokens=response.input_tokens,
        judge_output_tokens=response.output_tokens,
    )
