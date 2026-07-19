from pathlib import Path

import pytest

from rag_agent_platform.ingestion.mock import MockIngestionPipeline
from rag_agent_platform.storage.mock import MockDocumentRepository


@pytest.fixture
def ingestion() -> MockIngestionPipeline:
    return MockIngestionPipeline(MockDocumentRepository(), parent_chunk_size=20, child_chunk_size=8)


@pytest.mark.parametrize(
    ("filename", "content"),
    [("sample.txt", "TXT 示例内容"), ("sample.md", "# 标题\nMarkdown 内容")],
)
def test_supported_text_file_can_be_ingested(
    tmp_path: Path,
    ingestion: MockIngestionPipeline,
    filename: str,
    content: str,
) -> None:
    file_path = tmp_path / filename
    file_path.write_text(content, encoding="utf-8")

    result = ingestion.ingest(file_path)

    assert result.document.filename == filename
    assert result.document.status == "ready"
    assert result.parent_chunk_count >= 1
    assert result.child_chunk_count >= 1
    assert result.document in ingestion.list_documents()


def test_missing_file_fails(ingestion: MockIngestionPipeline, tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="does not exist"):
        ingestion.ingest(tmp_path / "missing.txt")


def test_empty_file_fails(ingestion: MockIngestionPipeline, tmp_path: Path) -> None:
    file_path = tmp_path / "empty.md"
    file_path.write_text("  \n", encoding="utf-8")

    with pytest.raises(ValueError, match="must not be empty"):
        ingestion.ingest(file_path)


def test_delete_removes_document(ingestion: MockIngestionPipeline, tmp_path: Path) -> None:
    file_path = tmp_path / "delete-me.txt"
    file_path.write_text("需要删除的文档", encoding="utf-8")
    result = ingestion.ingest(file_path)

    ingestion.delete_document(result.document.document_id)

    assert result.document not in ingestion.list_documents()
