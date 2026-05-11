import pytest

from backend.llm.mock_provider import MockLLMProvider
from backend.llm.validators import LLMMessage, LLMRequest


def test_mock_provider_returns_response() -> None:
    provider = MockLLMProvider()
    request = LLMRequest(messages=[LLMMessage(role="user", content="hello")])
    response = provider.generate(request)
    assert "hello" in response.text
    assert response.provider == "mock"
    assert response.output_tokens > 0


def test_mock_provider_stream_emits_done() -> None:
    provider = MockLLMProvider()
    request = LLMRequest(messages=[LLMMessage(role="user", content="hello world")])
    chunks = list(provider.stream(request))
    assert chunks[-1].done is True
    text = "".join(c.delta for c in chunks)
    assert "hello" in text


def test_mock_provider_canned_response() -> None:
    provider = MockLLMProvider(canned_response="fixed answer")
    request = LLMRequest(messages=[LLMMessage(role="user", content="anything")])
    assert provider.generate(request).text == "fixed answer"


@pytest.mark.asyncio
async def test_mock_provider_generate_async() -> None:
    provider = MockLLMProvider()
    request = LLMRequest(messages=[LLMMessage(role="user", content="hello")])
    response = await provider.generate_async(request)
    assert "hello" in response.text
