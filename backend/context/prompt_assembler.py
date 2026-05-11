from backend.context.validators import ContextBlock


def assemble_prompt(query: str, blocks: list[ContextBlock]) -> str:
    """Build a citation-numbered LLM prompt. Format is provider-neutral — the next phase
    can post-process for Anthropic / OpenAI / etc.
    """
    if not blocks:
        return ""

    sections: list[str] = []
    for index, block in enumerate(blocks, start=1):
        path = " > ".join(block.section_path) if block.section_path else block.title
        sections.append(
            f"[{index}] {path}\n"
            f"Source: {block.source_url}\n\n"
            f"{block.text.strip()}\n"
        )

    context_str = "\n---\n".join(sections)
    return (
        f"Question: {query}\n\n"
        f"Cite sources using [1], [2], ... matching the indices below. Only answer from "
        f"the provided context; if the context is insufficient, say so.\n\n"
        f"Context:\n\n"
        f"{context_str}\n"
    )
