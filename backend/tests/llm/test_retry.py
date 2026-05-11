import pytest

from backend.llm.retry import with_retry
from backend.llm.validators import ProviderError


def test_with_retry_succeeds_on_first_attempt() -> None:
    calls = []

    def fn() -> str:
        calls.append("call")
        return "ok"

    assert with_retry(fn) == "ok"
    assert len(calls) == 1


def test_with_retry_retries_on_provider_error() -> None:
    attempts: list[int] = []

    def fn() -> str:
        attempts.append(1)
        if len(attempts) < 3:
            raise ProviderError("test", "transient")
        return "ok"

    assert with_retry(fn, max_attempts=3, base_delay=0.0) == "ok"
    assert len(attempts) == 3


def test_with_retry_exhausts_and_raises() -> None:
    def fn() -> str:
        raise ProviderError("test", "always fails")

    with pytest.raises(ProviderError):
        with_retry(fn, max_attempts=2, base_delay=0.0)


def test_with_retry_does_not_catch_other_exceptions() -> None:
    def fn() -> str:
        raise ValueError("not a provider error")

    with pytest.raises(ValueError):
        with_retry(fn, max_attempts=3, base_delay=0.0)
