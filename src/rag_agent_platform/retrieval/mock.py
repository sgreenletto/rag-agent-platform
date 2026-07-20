"""Deterministic in-memory Retriever for contract testing."""

from rag_agent_platform.models import RetrievedChunk
from rag_agent_platform.retrieval.base import BaseRetriever
from rag_agent_platform.retrieval.validation import finalize_results, validate_retrieval_request


class MockRetriever(BaseRetriever):
    """Return deterministic sample chunks without embeddings or databases."""

    def __init__(
        self,
        retrieval_method: str,
        chunks: list[RetrievedChunk] | None = None,
    ) -> None:
        if not retrieval_method.strip():
            raise ValueError("retrieval_method must not be empty")
        self.retrieval_method = retrieval_method
        self._chunks = chunks if chunks is not None else self._default_chunks(retrieval_method)

    def retrieve(
        self,
        query: str,
        document_ids: list[str] | None = None,
        top_k: int = 5,
    ) -> list[RetrievedChunk]:
        validate_retrieval_request(query, top_k)
        return finalize_results(self._chunks, document_ids, top_k)

    @staticmethod
    def _default_chunks(retrieval_method: str) -> list[RetrievedChunk]:
        return [
            RetrievedChunk(
                chunk_id=f"{retrieval_method}-chunk-1",
                content=f"这是 {retrieval_method} 检索器返回的第一条 Mock 证据。",
                normalized_score=0.92,
                source="mock-policy.txt",
                document_id="mock-document-1",
                parent_id="mock-parent-1",
                page=1,
                retrieval_method=retrieval_method,
                metadata={"mock": True},
            ),
            RetrievedChunk(
                chunk_id=f"{retrieval_method}-chunk-2",
                content=f"这是 {retrieval_method} 检索器返回的第二条 Mock 证据。",
                normalized_score=0.81,
                source="mock-handbook.md",
                document_id="mock-document-2",
                parent_id="mock-parent-2",
                page=2,
                retrieval_method=retrieval_method,
                metadata={"mock": True},
            ),
            RetrievedChunk(
                chunk_id=f"{retrieval_method}-chunk-3",
                content=f"这是 {retrieval_method} 检索器返回的第三条 Mock 证据。",
                normalized_score=0.7,
                source="mock-policy.txt",
                document_id="mock-document-1",
                parent_id="mock-parent-3",
                page=3,
                retrieval_method=retrieval_method,
                metadata={"mock": True},
            ),
        ]
