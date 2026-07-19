"""Fixed-length TXT/Markdown ingestion for the Mock development loop."""

from pathlib import Path
from uuid import uuid4

from rag_agent_platform.ingestion.base import IngestionPipeline
from rag_agent_platform.models import (
    ChildChunk,
    DocumentRecord,
    IngestionResult,
    ParentChunk,
)
from rag_agent_platform.storage.base import DocumentRepository


class MockIngestionPipeline(IngestionPipeline):
    """Read local text files and create simple in-memory parent/child chunks."""

    _SUPPORTED_SUFFIXES = {".txt", ".md"}

    def __init__(
        self,
        repository: DocumentRepository,
        parent_chunk_size: int = 500,
        child_chunk_size: int = 200,
    ) -> None:
        if parent_chunk_size <= 0 or child_chunk_size <= 0:
            raise ValueError("chunk sizes must be greater than 0")
        self._repository = repository
        self._parent_chunk_size = parent_chunk_size
        self._child_chunk_size = child_chunk_size

    def ingest(self, file_path: str | Path) -> IngestionResult:
        path = Path(file_path)
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"document does not exist: {path}")
        if path.suffix.lower() not in self._SUPPORTED_SUFFIXES:
            raise ValueError(
                f"unsupported file type: {path.suffix or '(none)'}; only .txt and .md are supported"
            )

        content = path.read_text(encoding="utf-8")
        if not content.strip():
            raise ValueError("document must not be empty")

        document_id = str(uuid4())
        document = DocumentRecord(
            document_id=document_id,
            filename=path.name,
            file_type=path.suffix.lower().lstrip("."),
            source_path=str(path.resolve()),
            status="ready",
            metadata={"mock": True},
        )
        parent_chunks, child_chunks = self._split(document_id, content)

        self._repository.save_document(document)
        self._repository.save_parent_chunks(parent_chunks)
        self._repository.save_child_chunks(child_chunks)
        return IngestionResult(
            document=document,
            parent_chunk_count=len(parent_chunks),
            child_chunk_count=len(child_chunks),
            warnings=["当前使用固定长度 Mock 切块，尚未进行真实文档解析。"],
        )

    def list_documents(self) -> list[DocumentRecord]:
        return self._repository.list_documents()

    def delete_document(self, document_id: str) -> None:
        self._repository.delete_document(document_id)

    def _split(self, document_id: str, content: str) -> tuple[list[ParentChunk], list[ChildChunk]]:
        parents: list[ParentChunk] = []
        children: list[ChildChunk] = []
        parent_texts = self._fixed_chunks(content, self._parent_chunk_size)
        for parent_index, parent_text in enumerate(parent_texts, start=1):
            parent_id = str(uuid4())
            parents.append(
                ParentChunk(
                    chunk_id=parent_id,
                    document_id=document_id,
                    content=parent_text,
                    metadata={"mock": True, "position": parent_index},
                )
            )
            for child_index, child_text in enumerate(
                self._fixed_chunks(parent_text, self._child_chunk_size), start=1
            ):
                children.append(
                    ChildChunk(
                        chunk_id=str(uuid4()),
                        document_id=document_id,
                        parent_id=parent_id,
                        content=child_text,
                        metadata={"mock": True, "position": child_index},
                    )
                )
        return parents, children

    @staticmethod
    def _fixed_chunks(content: str, chunk_size: int) -> list[str]:
        return [content[start : start + chunk_size] for start in range(0, len(content), chunk_size)]
