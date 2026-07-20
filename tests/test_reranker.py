import pytest

from rag_agent_platform.models import RetrievedChunk
from rag_agent_platform.retrieval.base import BaseRetriever
from rag_agent_platform.retrieval.reranker import (
    BaseReranker,
    RerankingRetriever,
    TokenOverlapReranker,
)


def result(chunk_id: str, content: str, score: float) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        content=content,
        normalized_score=score,
        source="source.txt",
        document_id="doc",
        retrieval_method="hybrid_rrf",
    )


class RecordingRetriever(BaseRetriever):
    def __init__(self, results: list[RetrievedChunk]) -> None:
        self.results = results
        self.top_k: int | None = None
        self.query: str | None = None
        self.document_ids: list[str] | None = None

    def retrieve(
        self,
        query: str,
        document_ids: list[str] | None = None,
        top_k: int = 5,
    ) -> list[RetrievedChunk]:
        self.query = query
        self.document_ids = document_ids
        self.top_k = top_k
        return self.results[:top_k]


class ReverseReranker(BaseReranker):
    def rerank(self, query: str, chunks: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        return list(reversed(chunks))[:top_k]


class InjectingReranker(BaseReranker):
    def rerank(self, query: str, chunks: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        return [result("injected", "not retrieved", 1.0)]


def test_token_overlap_reranker_promotes_lexically_relevant_candidate() -> None:
    chunks = [
        result("high-recall", "差旅费用规定", 1.0),
        result("relevant", "员工年假申请流程", 0.5),
    ]

    reranked = TokenOverlapReranker(overlap_weight=0.8).rerank("年假申请", chunks, top_k=2)

    assert [chunk.chunk_id for chunk in reranked] == ["relevant", "high-recall"]
    assert reranked[0].metadata["pre_rerank_score"] == 0.5
    assert reranked[0].metadata["token_overlap_score"] == 1.0
    assert reranked[0].retrieval_method.endswith("+token_rerank")


def test_reranking_retriever_expands_candidates_before_reranking() -> None:
    base = RecordingRetriever([result("a", "无关", 1.0), result("b", "年假制度", 0.8)])
    retriever = RerankingRetriever(base, TokenOverlapReranker(), candidate_multiplier=5)

    reranked = retriever.retrieve("年假", top_k=1)

    assert base.top_k == 5
    assert [chunk.chunk_id for chunk in reranked] == ["b"]


def test_token_overlap_does_not_mutate_input_chunk() -> None:
    original = result("a", "年假制度", 0.4)

    reranked = TokenOverlapReranker().rerank("年假", [original], top_k=1)

    assert original.normalized_score == 0.4
    assert original.metadata == {}
    assert reranked[0] is not original


def test_token_overlap_with_zero_weight_preserves_retrieval_ranking() -> None:
    chunks = [result("a", "无关", 0.9), result("b", "年假", 0.2)]

    reranked = TokenOverlapReranker(overlap_weight=0.0).rerank("年假", chunks, top_k=2)

    assert [chunk.chunk_id for chunk in reranked] == ["a", "b"]
    assert [chunk.normalized_score for chunk in reranked] == [0.9, 0.2]


def test_token_overlap_returns_empty_for_empty_candidates() -> None:
    assert TokenOverlapReranker().rerank("query", [], top_k=3) == []


@pytest.mark.parametrize("weight", [-0.01, 1.01])
def test_token_overlap_rejects_invalid_weight(weight: float) -> None:
    with pytest.raises(ValueError, match="overlap_weight"):
        TokenOverlapReranker(overlap_weight=weight)


@pytest.mark.parametrize(("query", "top_k"), [("  ", 1), ("query", 0)])
def test_token_overlap_validates_request(query: str, top_k: int) -> None:
    with pytest.raises(ValueError):
        TokenOverlapReranker().rerank(query, [], top_k=top_k)


def test_reranking_retriever_forwards_normalized_query_and_document_filter() -> None:
    base = RecordingRetriever([result("a", "年假", 0.8)])
    retriever = RerankingRetriever(base, ReverseReranker(), candidate_multiplier=2)

    retriever.retrieve("  年假  ", document_ids=["doc"], top_k=3)

    assert base.query == "年假"
    assert base.document_ids == ["doc"]
    assert base.top_k == 6


def test_reranking_retriever_accepts_custom_reranker() -> None:
    base = RecordingRetriever([result("first", "一", 1.0), result("second", "二", 0.5)])

    reranked = RerankingRetriever(base, ReverseReranker()).retrieve("query", top_k=1)

    assert [chunk.chunk_id for chunk in reranked] == ["second"]


def test_reranking_retriever_rejects_invalid_candidate_multiplier() -> None:
    with pytest.raises(ValueError, match="candidate_multiplier"):
        RerankingRetriever(RecordingRetriever([]), ReverseReranker(), candidate_multiplier=0)


def test_reranking_retriever_rejects_injected_evidence() -> None:
    retriever = RerankingRetriever(
        RecordingRetriever([result("real", "retrieved", 0.8)]), InjectingReranker()
    )

    with pytest.raises(ValueError, match="unknown chunk_id"):
        retriever.retrieve("query")
