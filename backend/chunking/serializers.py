from pathlib import Path

import orjson

from backend.chunking.models.chunk import SemanticChunk


class ChunkSerializer:
    def write_jsonl(self, chunks: list[SemanticChunk], output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("wb") as file:
            for chunk in chunks:
                file.write(orjson.dumps(chunk.model_dump(mode="json")))
                file.write(b"\n")

