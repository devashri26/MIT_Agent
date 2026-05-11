from backend.context.validators import Citation
from backend.reranking.validators import RerankedChunk


def build_citation(chunk: RerankedChunk) -> Citation:
    return Citation(
        chunk_id=chunk.chunk_id,
        source_url=chunk.url,
        title=chunk.title,
        section_path=list(chunk.section_path or []),
    )
