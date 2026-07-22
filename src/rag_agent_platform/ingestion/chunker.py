"""Parent/child chunking boundary for document ingestion."""

from dataclasses import dataclass

from rag_agent_platform.models import ChildChunk, ParentChunk


@dataclass(slots=True)
class ChunkingConfig:
    """Configuration for parent/child chunk generation."""

    parent_chunk_size: int = 1000
    child_chunk_size: int = 400
    child_overlap: int = 80

    def __post_init__(self) -> None:
        if self.parent_chunk_size <= 0:
            raise ValueError("parent_chunk_size must be greater than 0")
        if self.child_chunk_size <= 0:
            raise ValueError("child_chunk_size must be greater than 0")
        if self.child_overlap < 0:
            raise ValueError("child_overlap must not be negative")
        if self.child_overlap >= self.child_chunk_size:
            raise ValueError("child_overlap must be smaller than child_chunk_size")


class ParentChildChunker:
    """Create retrievable child chunks and larger parent chunks."""

    def __init__(self, config: ChunkingConfig | None = None) -> None:
        self.config = config or ChunkingConfig()

    def split(
        self,
        *,
        document_id: str,
        content: str,
        source: str,
        file_type: str,
    ) -> tuple[list[ParentChunk], list[ChildChunk]]:
        """Split text into parent and child chunks."""
        normalized_content = content.strip()
        if not normalized_content:
            return [], []

        parents: list[ParentChunk] = []
        children: list[ChildChunk] = []
        parent_texts = self._fixed_chunks(normalized_content, self.config.parent_chunk_size)
        for parent_index, parent_text in enumerate(parent_texts, start=1):
            parent_id = f"{document_id}:parent:{parent_index}"
            parent = ParentChunk(
                chunk_id=parent_id,
                document_id=document_id,
                content=parent_text,
                metadata={
                    "document_id": document_id,
                    "chunk_id": parent_id,
                    "source": source,
                    "file_type": file_type,
                    "position": parent_index,
                },
            )
            parents.append(parent)

            child_texts = self._overlap_chunks(
                parent_text,
                chunk_size=self.config.child_chunk_size,
                overlap=self.config.child_overlap,
            )
            for child_index, child_text in enumerate(child_texts, start=1):
                child_id = f"{document_id}:child:{parent_index}:{child_index}"
                children.append(
                    ChildChunk(
                        chunk_id=child_id,
                        document_id=document_id,
                        parent_id=parent_id,
                        content=child_text,
                        metadata={
                            "document_id": document_id,
                            "parent_id": parent_id,
                            "chunk_id": child_id,
                            "source": source,
                            "file_type": file_type,
                            "page": None,
                            "parent_position": parent_index,
                            "position": child_index,
                        },
                    )
                )
        return parents, children

    @staticmethod
    def _fixed_chunks(text: str, chunk_size: int) -> list[str]:
        return [
            text[start : start + chunk_size].strip()
            for start in range(0, len(text), chunk_size)
            if text[start : start + chunk_size].strip()
        ]

    @staticmethod
    def _overlap_chunks(text: str, chunk_size: int, overlap: int) -> list[str]:
        step = chunk_size - overlap
        chunks: list[str] = []
        start = 0
        while start < len(text):
            chunk = text[start : start + chunk_size].strip()
            if chunk:
                chunks.append(chunk)
            if start + chunk_size >= len(text):
                break
            start += step
        return chunks
