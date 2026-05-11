GENERIC_HEADINGS = {
    "overview",
    "introduction",
    "home",
    "general",
    "about",
    "welcome",
    "main",
    "index",
    "page",
    "header",
    "footer",
    "content",
    "details",
    "more",
    "info",
    "information",
    "read more",
    "click here",
}


def is_generic_heading(heading: str) -> bool:
    if not heading:
        return True
    normalized = heading.strip().lower()
    if not normalized:
        return True
    if normalized in GENERIC_HEADINGS:
        return True
    words = [w for w in normalized.split() if w]
    return bool(words) and all(w in GENERIC_HEADINGS for w in words)
