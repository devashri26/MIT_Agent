from backend.normalization.heading_classifier import is_generic_heading
from backend.normalization.hierarchy_extractor import (
    clean_document_title,
    humanize_page_type,
    humanize_section_type,
)


def normalize_section_path(
    page_type: str,
    section_type: str,
    document_title: str,
    metadata_headings: list[str] | None = None,
) -> list[str]:
    """Synthesize a section_path from Phase 2 metadata since the chunker did not preserve
    h1/h2/h3 hierarchy. Filters generic headings; dedupes case-insensitive while preserving
    order (broadest → most specific)."""
    candidates: list[str] = []

    page_label = humanize_page_type(page_type)
    if page_label and not is_generic_heading(page_label):
        candidates.append(page_label)

    section_label = humanize_section_type(section_type)
    if section_label and not is_generic_heading(section_label):
        candidates.append(section_label)

    title = clean_document_title(document_title)
    if title and not is_generic_heading(title):
        candidates.append(title)

    for heading in metadata_headings or []:
        if heading and not is_generic_heading(heading):
            candidates.append(heading)

    path: list[str] = []
    seen: set[str] = set()
    for item in candidates:
        key = item.strip().lower()
        if not key or key in seen:
            continue
        path.append(item.strip())
        seen.add(key)
    return path


def needs_hierarchy_repair(section_heading: str) -> bool:
    return is_generic_heading(section_heading)
