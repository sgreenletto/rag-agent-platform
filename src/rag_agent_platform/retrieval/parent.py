"""Parent-context adapter for child-chunk retrievers."""

from dataclasses import replace
from typing import Protocol, runtime_checkable

from rag_agent_platform.models import ParentChunk, RetrievedChunk
from rag_agent_platform.retrieval.base import BaseRetriever
from rag_agent_platform.retrieval.validation import finalize_results, validate_retrieval_request


@runtime_checkable
class ParentChunkLookup(Protocol):
    def get_parent_chunk(self, chunk_id: str) -> ParentChunk | None:
        """Return one stored parent chunk."""
        ...


class ParentContextRetriever(BaseRetriever):
    """Replace matched child content with its stored parent while preserving provenance."""

    def __init__(self, retriever: BaseRetriever, repository: ParentChunkLookup) -> None:
        self._retriever = retriever
        self._repository = repository

    def retrieve(
        self,
        query: str,
        document_ids: list[str] | None = None,
        top_k: int = 5,
    ) -> list[RetrievedChunk]:
        normalized_query = validate_retrieval_request(query, top_k)
        children = self._retriever.retrieve(normalized_query, document_ids, top_k)
        expanded: list[RetrievedChunk] = []
        for child in children:
            parent = self._repository.get_parent_chunk(child.parent_id) if child.parent_id else None
            if parent is None:
                expanded.append(child)
                continue
            metadata = {
                **child.metadata,
                "matched_child_content": child.content,
                "parent_context_chunk_id": parent.chunk_id,
            }
            expanded.append(
                replace(
                    child,
                    content=parent.content,
                    retrieval_method=f"{child.retrieval_method}+parent",
                    metadata=metadata,
                )
            )
        return finalize_results(expanded, document_ids, top_k)
