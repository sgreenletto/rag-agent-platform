"""Parent/child chunking boundary for document ingestion."""

from dataclasses import dataclass

from rag_agent_platform.models import ChildChunk, ParentChunk


@dataclass(slots=True)
class ChunkingConfig:
    """Configuration for parent/child chunk generation."""

    parent_chunk_size: int = 1000
    child_chunk_size: int = 400
    child_overlap: int = 80


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
        """Split text into parent and child chunks.

        Real splitting is implemented in the parent/child chunking step.
        """
        raise NotImplementedError("parent/child chunking is not implemented yet")
