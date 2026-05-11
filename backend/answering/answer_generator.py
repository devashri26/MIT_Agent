from __future__ import annotations

from backend.context.validators import GroundedContext
from backend.llm.prompts.grounded_answering import SYSTEM_PROMPT, build_user_message
from backend.llm.provider_interface import BaseLLMProvider
from backend.llm.validators import LLMMessage, LLMRequest


def generate_answer(
    provider: BaseLLMProvider,
    query: str,
    grounded_context: GroundedContext,
    model: str = "",
    temperature: float = 0.0,
    max_tokens: int = 1500,
) -> tuple[str, str, str, int, int]:
    """Synchronous LLM call. Returns (answer_text, provider_name, model_used,
    input_tokens, output_tokens). Token counts let callers surface usage to the UI."""
    user_message = build_user_message(query, grounded_context.prompt)
    request = LLMRequest(
        system_prompt=SYSTEM_PROMPT,
        messages=[LLMMessage(role="user", content=user_message)],
        model=model or provider.default_model,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    response = provider.generate(request)
    return response.text, response.provider, response.model, response.input_tokens, response.output_tokens
