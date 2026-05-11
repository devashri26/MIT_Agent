from typing import Any


SKIP_FLAGS = {"cta_heavy", "boilerplate_heavy"}


def is_embedding_eligible(chunk: dict[str, Any]) -> bool:
    """Per spec: skip reusable components, contaminated chunks, CTA-heavy, boilerplate-heavy."""
    if chunk.get("is_reusable_component"):
        return False
    if chunk.get("cross_domain_contamination"):
        return False
    flags = set(chunk.get("quality_flags") or [])
    if flags & SKIP_FLAGS:
        return False
    return True


def skip_reason(chunk: dict[str, Any]) -> str | None:
    if chunk.get("is_reusable_component"):
        return "reusable_component"
    if chunk.get("cross_domain_contamination"):
        return "cross_domain_contamination"
    flags = set(chunk.get("quality_flags") or [])
    for flag in ("cta_heavy", "boilerplate_heavy"):
        if flag in flags:
            return flag
    return None
