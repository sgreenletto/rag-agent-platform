"""GraphRetriever — adapts a GraphService to the BaseRetriever contract.

This adapter allows the GraphRAG pipeline to be injected anywhere a
``BaseRetriever`` is expected (e.g. ``MockAgentService``, the UI
assembly layer, or evaluation harnesses).
"""

from __future__ import annotations

from rag_agent_platform.graph.base import GraphService
from rag_agent_platform.retrieval.base import BaseRetriever

try:
    from rag_agent_platform.models import RetrievedChunk
except ImportError:  # pragma: no cover
    RetrievedChunk = None  # type: ignore[assignment]


class GraphRetriever(BaseRetriever):
    """Expose graph-flavoured retrieval through the standard
    ``BaseRetriever`` interface.

    The retriever delegates to an underlying :class:`GraphService` and
    enforces the same input-validation semantics as
    :class:`MockRetriever`.

    Parameters:
        graph_service: A concrete ``GraphService`` (e.g.
            ``NetworkXGraphService``) that already has a built graph.
        retrieval_method: Value stamped into
            ``RetrievedChunk.retrieval_method`` (default ``"graph"``).
    """

    def __init__(
        self,
        graph_service: GraphService,
        retrieval_method: str = "graph",
    ) -> None:
        if not isinstance(graph_service, GraphService):
            raise TypeError(
                f"graph_service must be a GraphService, got {type(graph_service).__name__}"
            )
        if not retrieval_method.strip():
            raise ValueError("retrieval_method must not be empty")

        self._graph_service = graph_service
        self._retrieval_method = retrieval_method

    # ------------------------------------------------------------------
    # BaseRetriever contract
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        document_ids: list[str] | None = None,
        top_k: int = 5,
    ) -> list[object]:
        """Retrieve graph evidence as ``list[RetrievedChunk]``.

        Args:
            query: Natural-language query string.  Entities are
                extracted from the query and used to traverse the
                knowledge graph.
            document_ids: Optional allow-list of document ids.
                ``None`` means all documents; ``[]`` means none.
            top_k: Maximum number of chunks to return.  Must be > 0.

        Returns:
            Chunks sorted by ``normalized_score`` descending.

        Raises:
            ValueError: If *query* is empty or *top_k* <= 0.
        """
        if not query.strip():
            raise ValueError("query must not be empty")
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        # Handle empty document_ids list → no results
        if document_ids is not None and len(document_ids) == 0:
            return []

        chunks = self._graph_service.retrieve(
            query=query,
            document_ids=document_ids,
            top_k=top_k,
        )

        # Stamp retrieval_method on every chunk
        for chunk in chunks:
            if hasattr(chunk, "retrieval_method"):
                chunk.retrieval_method = self._retrieval_method

        # Ensure ordering
        chunks.sort(key=lambda c: getattr(c, "normalized_score", 0.0), reverse=True)
        return chunks[:top_k]
