from pathlib import Path

from pydantic import BaseModel, Field


class Settings(BaseModel):
    processed_output_path: Path = Field(
        default=Path("datasets/processed_documents.json"),
        description="Default JSON export path for processed website documents.",
    )
    processed_ndjson_output_path: Path = Field(default=Path("datasets/processed_documents.ndjson"))
    reports_dir: Path = Field(default=Path("reports"))
    csv_chunk_size: int = Field(default=1_000, gt=0)


settings = Settings()
