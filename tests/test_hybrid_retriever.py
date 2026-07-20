import pytest

from rag_agent_platform.models import RetrievedChunk
from rag_agent_platform.retrieval.base import BaseRetriever
from rag_agent_platform.retrieval.hybrid import HybridRetriever


def result(chunk_id: str, document_id: str, score: float, method: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        content=f"content-{chunk_id}",
        normalized_score=score,
        source=f"{document_id}.txt",
        document_id=document_id,
        retrieval_method=method,
    )


class RecordingRetriever(BaseRetriever):
    def __init__(self, results: list[RetrievedChunk]) -> None:
        self.results = results
        self.calls: list[tuple[str, list[str] | None, int]] = []

    def retrieve(
        self,
        query: str,
        document_ids: list[str] | None = None,
        top_k: int = 5,
    ) -> list[RetrievedChunk]:
        self.calls.append((query, document_ids, top_k))
        return self.results[:top_k]


class FailingRetriever(BaseRetriever):
    def retrieve(
        self,
        query: str,
        document_ids: list[str] | None = None,
        top_k: int = 5,
    ) -> list[RetrievedChunk]:
        raise ConnectionError("backend unavailable")


def test_hybrid_expands_candidates_fuses_and_applies_top_k() -> None:
    dense = RecordingRetriever(
        [result("dense", "doc", 1.0, "dense"), result("both", "doc", 0.8, "dense")]
    )
    sparse = RecordingRetriever(
        [result("both", "doc", 1.0, "bm25"), result("sparse", "doc", 0.7, "bm25")]
    )
    retriever = HybridRetriever(dense, sparse, candidate_multiplier=4)

    fused = retriever.retrieve("  query  ", document_ids=["doc"], top_k=2)

    assert fused[0].chunk_id == "both"
    assert len(fused) == 2
    assert dense.calls == [("query", ["doc"], 8)]
    assert sparse.calls == [("query", ["doc"], 8)]


def test_hybrid_fallback_is_visible_in_result_metadata() -> None:
    sparse = RecordingRetriever([result("sparse", "doc", 1.0, "bm25")])

    fused = HybridRetriever(FailingRetriever(), sparse).retrieve("query")

    assert fused[0].chunk_id == "sparse"
    assert "dense" in fused[0].metadata["retrieval_warnings"]


def test_hybrid_raises_when_all_retrievers_fail() -> None:
    retriever = HybridRetriever(FailingRetriever(), FailingRetriever())

    with pytest.raises(RuntimeError, match="all hybrid retrievers failed"):
        retriever.retrieve("query")


def test_hybrid_strict_mode_propagates_failure() -> None:
    retriever = HybridRetriever(FailingRetriever(), RecordingRetriever([]), failure_mode="raise")

    with pytest.raises(ConnectionError, match="backend unavailable"):
        retriever.retrieve("query")
