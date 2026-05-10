from pathlib import Path

from backend.chunking.chunk_pipeline import ChunkPipeline


def main() -> None:
    report = ChunkPipeline().run(
        input_path=Path("datasets/processed_documents.json"),
        output_path=Path("datasets/chunks.jsonl"),
        report_path=Path("reports/chunking_report.json"),
    )
    print(report.model_dump_json())


if __name__ == "__main__":
    main()

