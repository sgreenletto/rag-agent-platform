"""Dataclass schemas forming the public boundaries between team modules."""

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class QueryType(StrEnum):
    """Coarse query intent used by the agent router."""

    SIMPLE = "simple"
    COMPLEX = "complex"
    RELATION = "relation"
    CHAT = "chat"


class RetrievalStrategy(StrEnum):
    """Retriever family selected for one request."""

    NONE = "none"
    NAIVE = "naive"
    ADVANCED = "advanced"
    GRAPH = "graph"


def _require_text(value: str, field_name: str) -> None:
    if not value.strip():
        raise ValueError(f"{field_name} must not be empty")


@dataclass(slots=True)
class DocumentRecord:
    document_id: str
    filename: str
    file_type: str
    source_path: str
    status: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ParentChunk:
    chunk_id: str
    document_id: str
    content: str
    page: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_text(self.content, "content")


@dataclass(slots=True)
class ChildChunk:
    chunk_id: str
    document_id: str
    parent_id: str
    content: str
    page: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_text(self.content, "content")


@dataclass(slots=True)
class RetrievedChunk:
    """Normalized evidence returned by every retrieval implementation."""

    chunk_id: str
    content: str
    normalized_score: float
    source: str
    document_id: str | None = None
    parent_id: str | None = None
    page: int | None = None
    retrieval_method: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_text(self.content, "content")
        _require_text(self.source, "source")
        if not 0.0 <= self.normalized_score <= 1.0:
            raise ValueError("normalized_score must be between 0.0 and 1.0")


@dataclass(slots=True)
class Citation:
    index: int
    source: str
    page: int | None = None
    chunk_id: str | None = None


@dataclass(slots=True)
class IngestionResult:
    document: DocumentRecord
    parent_chunk_count: int
    child_chunk_count: int
    warnings: list[str] = field(default_factory=list)


@dataclass(slots=True)
class AgentResult:
    answer: str
    citations: list[Citation]
    strategy: RetrievalStrategy
    query_type: QueryType
    retry_count: int
    execution_trace: list[str] = field(default_factory=list)
    retrieved_chunks: list[RetrievedChunk] = field(default_factory=list)
    error: str | None = None
    regenerate_count: int = 0
    refused: bool = False
    evaluation_decision: str | None = None
    query_history: list[str] = field(default_factory=list)
    strategy_history: list[RetrievalStrategy] = field(default_factory=list)
