"""Named adapters around member-two retriever compositions."""

from rag_agent_platform.models import RetrievedChunk
from rag_agent_platform.retrieval.base import BaseRetriever


class _DelegatingRetriever(BaseRetriever):
    def __init__(self, retriever: BaseRetriever) -> None:
        self._retriever = retriever

    def retrieve(
        self,
        query: str,
        document_ids: list[str] | None = None,
        top_k: int = 5,
    ) -> list[RetrievedChunk]:
        return self._retriever.retrieve(query, document_ids=document_ids, top_k=top_k)


class NaiveRetriever(_DelegatingRetriever):
    """Named Naive RAG adapter around dense retrieval and parent lookup."""


class AdvancedRetriever(_DelegatingRetriever):
    """Named adapter around the existing hybrid/RRF/rerank/compression pipeline."""
