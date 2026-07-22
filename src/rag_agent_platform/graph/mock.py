"""Deterministic Mock GraphRetriever for contract testing.

Mirrors the design of ``MockRetriever`` — no graph store, no
extractor, just pre-canned ``RetrievedChunk`` objects tagged with
``retrieval_method="graph"``.
"""

from __future__ import annotations

from rag_agent_platform.retrieval.base import BaseRetriever

try:
    from rag_agent_platform.models import RetrievedChunk
except ImportError:  # pragma: no cover
    RetrievedChunk = None  # type: ignore[assignment]


class MockGraphRetriever(BaseRetriever):
    """Return deterministic sample chunks without a graph backend.

    Useful for:
    - Contract tests that verify ``BaseRetriever`` compliance
    - UI smoke tests that need any ``graph`` retriever wired in
    - Agent routing tests (``mode="graph"`` → ``RetrievalStrategy.GRAPH``)
    """

    def __init__(
        self,
        retrieval_method: str = "graph",
        chunks: list[object] | None = None,
    ) -> None:
        if not retrieval_method.strip():
            raise ValueError("retrieval_method must not be empty")
        self.retrieval_method = retrieval_method
        self._chunks = (
            list(chunks) if chunks is not None else self._default_chunks(retrieval_method)
        )

    # ------------------------------------------------------------------
    # BaseRetriever contract
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        document_ids: list[str] | None = None,
        top_k: int = 5,
    ) -> list[object]:
        """Return deterministic graph mock chunks."""
        if not query.strip():
            raise ValueError("query must not be empty")
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        allowed_ids = set(document_ids) if document_ids else None
        filtered = [
            chunk
            for chunk in self._chunks
            if allowed_ids is None or getattr(chunk, "document_id", None) in allowed_ids
        ]
        filtered.sort(
            key=lambda chunk: getattr(chunk, "normalized_score", 0.0),
            reverse=True,
        )
        return filtered[:top_k]

    # ------------------------------------------------------------------
    # Default sample data
    # ------------------------------------------------------------------

    @staticmethod
    def _default_chunks(retrieval_method: str) -> list[object]:
        """Pre-built graph-style chunks used when none are provided."""
        return [
            RetrievedChunk(
                chunk_id=f"{retrieval_method}-chunk-1",
                content=f"这是 {retrieval_method} 知识图谱检索返回的第一条证据。"
                "实体关系：人事部 —[负责]→ 请假制度。",
                normalized_score=0.93,
                source="mock-graph-policy.txt",
                document_id="mock-document-1",
                parent_id="mock-parent-1",
                page=1,
                retrieval_method=retrieval_method,
                metadata={"mock": True, "entities": ["人事部", "请假制度"]},
            ),
            RetrievedChunk(
                chunk_id=f"{retrieval_method}-chunk-2",
                content=f"这是 {retrieval_method} 知识图谱检索返回的第二条证据。"
                "实体关系：请假制度 —[属于]→ 公司规章。",
                normalized_score=0.84,
                source="mock-graph-handbook.md",
                document_id="mock-document-2",
                parent_id="mock-parent-2",
                page=2,
                retrieval_method=retrieval_method,
                metadata={"mock": True, "entities": ["请假制度", "公司规章"]},
            ),
            RetrievedChunk(
                chunk_id=f"{retrieval_method}-chunk-3",
                content=f"这是 {retrieval_method} 知识图谱检索返回的第三条证据。"
                "实体关系：员工 —[依赖]→ 考勤系统 → 请假流程。",
                normalized_score=0.76,
                source="mock-graph-policy.txt",
                document_id="mock-document-1",
                parent_id="mock-parent-3",
                page=3,
                retrieval_method=retrieval_method,
                metadata={"mock": True, "entities": ["员工", "考勤系统", "请假流程"]},
            ),
            RetrievedChunk(
                chunk_id=f"{retrieval_method}-chunk-4",
                content=f"这是 {retrieval_method} 知识图谱检索返回的第四条多跳证据。"
                "实体关系：部门经理 —[审批]→ 请假申请 —[触发]→ 薪资计算。",
                normalized_score=0.68,
                source="mock-graph-handbook.md",
                document_id="mock-document-2",
                parent_id="mock-parent-4",
                page=4,
                retrieval_method=retrieval_method,
                metadata={
                    "mock": True,
                    "entities": ["部门经理", "请假申请", "薪资计算"],
                    "hops": 2,
                },
            ),
        ]
