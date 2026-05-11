from __future__ import annotations

import time
from typing import Callable, TypeVar

from backend.llm.validators import ProviderError


T = TypeVar("T")


def with_retry(
    fn: Callable[[], T],
    *,
    max_attempts: int = 3,
    base_delay: float = 0.5,
    backoff: float = 2.0,
) -> T:
    """Tiny exponential-backoff retry helper. Only ProviderError triggers a retry; other
    exceptions propagate immediately so test failures aren't masked."""
    last_error: Exception | None = None
    delay = base_delay
    for attempt in range(1, max_attempts + 1):
        try:
            return fn()
        except ProviderError as exc:
            last_error = exc
            if attempt == max_attempts:
                raise
            time.sleep(delay)
            delay *= backoff
    raise ProviderError("retry", "exhausted attempts", last_error)
