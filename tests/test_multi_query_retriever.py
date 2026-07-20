import pytest

from rag_agent_platform.models import RetrievedChunk
from rag_agent_platform.retrieval.base import BaseRetriever
from rag_agent_platform.retrieval.multi_query import (
    IdentityQueryTransformer,
    MultiQueryRetriever,
    QueryTransformer,
)


def result(chunk_id: str, score: float = 1.0) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        content=f"content-{chunk_id}",
        normalized_score=score,
        source="source.txt",
        document_id="doc",
        retrieval_method="fixture",
    )


class StaticTransformer:
    def __init__(self, rewrites: list[str]) -> None:
        self.rewrites = rewrites

    def transform(self, query: str) -> list[str]:
        return self.rewrites


class FailingTransformer:
    def transform(self, query: str) -> list[str]:
        raise RuntimeError("rewrite service unavailable")


class MalformedTransformer:
    def transform(self, query: str) -> list[str]:
        return None  # type: ignore[return-value]


class QueryAwareRetriever(BaseRetriever):
    def __init__(self, results: dict[str, list[RetrievedChunk]]) -> None:
        self.results = results
        self.calls: list[tuple[str, list[str] | None, int]] = []

    def retrieve(
        self,
        query: str,
        document_ids: list[str] | None = None,
        top_k: int = 5,
    ) -> list[RetrievedChunk]:
        self.calls.append((query, document_ids, top_k))
        allowed = set(document_ids) if document_ids else None
        return [
            chunk
            for chunk in self.results.get(query, [])
            if allowed is None or chunk.document_id in allowed
        ][:top_k]


class FailingRetriever(BaseRetriever):
    def retrieve(
        self,
        query: str,
        document_ids: list[str] | None = None,
        top_k: int = 5,
    ) -> list[RetrievedChunk]:
        raise ConnectionError("retrieval failed")


def test_multi_query_keeps_original_deduplicates_and_limits_rewrites() -> None:
    base = QueryAwareRetriever(
        {
            "原问题": [result("original"), result("shared", 0.8)],
            "改写一": [result("shared"), result("rewrite")],
        }
    )
    retriever = MultiQueryRetriever(
        base,
        StaticTransformer(["原问题", "  ", "改写一", "改写二"]),
        max_queries=2,
        candidate_multiplier=3,
    )

    fused = retriever.retrieve(" 原问题 ", top_k=2)

    assert fused[0].chunk_id == "shared"
    assert base.calls == [("原问题", None, 6), ("改写一", None, 6)]
    assert fused[0].metadata["query_variants"] == {
        "query_0": "原问题",
        "query_1": "改写一",
    }


def test_multi_query_falls_back_to_original_when_transformer_fails() -> None:
    base = QueryAwareRetriever({"query": [result("original")]})

    fused = MultiQueryRetriever(base, FailingTransformer()).retrieve("query")

    assert fused[0].chunk_id == "original"
    assert "RuntimeError" in fused[0].metadata["query_transform_warning"]


def test_multi_query_strict_transform_mode_propagates_error() -> None:
    retriever = MultiQueryRetriever(
        QueryAwareRetriever({}), FailingTransformer(), fallback_on_transform_error=False
    )

    with pytest.raises(RuntimeError, match="rewrite service unavailable"):
        retriever.retrieve("query")


def test_identity_transformer_satisfies_runtime_protocol() -> None:
    transformer = IdentityQueryTransformer()

    assert isinstance(transformer, QueryTransformer)
    assert transformer.transform("query") == ["query"]


def test_multi_query_defaults_to_one_original_query() -> None:
    base = QueryAwareRetriever({"query": [result("original")]})

    fused = MultiQueryRetriever(base).retrieve("query", top_k=2)

    assert [chunk.chunk_id for chunk in fused] == ["original"]
    assert base.calls == [("query", None, 4)]


def test_multi_query_forwards_document_filter_to_every_variant() -> None:
    base = QueryAwareRetriever({"query": [result("original")]})
    retriever = MultiQueryRetriever(base, StaticTransformer(["rewrite"]))

    retriever.retrieve("query", document_ids=["doc"], top_k=1)

    assert base.calls == [("query", ["doc"], 2), ("rewrite", ["doc"], 2)]


def test_multi_query_returns_empty_when_all_variants_have_no_results() -> None:
    retriever = MultiQueryRetriever(QueryAwareRetriever({}), StaticTransformer(["rewrite"]))

    assert retriever.retrieve("query") == []


def test_multi_query_falls_back_for_malformed_transformer_output() -> None:
    base = QueryAwareRetriever({"query": [result("original")]})

    fused = MultiQueryRetriever(base, MalformedTransformer()).retrieve("query")

    assert [chunk.chunk_id for chunk in fused] == ["original"]
    assert "list[str]" in fused[0].metadata["query_transform_warning"]


def test_multi_query_does_not_hide_retrieval_failures() -> None:
    retriever = MultiQueryRetriever(FailingRetriever(), StaticTransformer([]))

    with pytest.raises(ConnectionError, match="retrieval failed"):
        retriever.retrieve("query")


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_queries": 0}, "max_queries"),
        ({"candidate_multiplier": 0}, "candidate_multiplier"),
        ({"rrf_k": 0}, "rrf_k"),
    ],
)
def test_multi_query_rejects_invalid_configuration(kwargs: dict[str, int], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        MultiQueryRetriever(QueryAwareRetriever({}), **kwargs)


@pytest.mark.parametrize(("query", "top_k"), [("  ", 1), ("query", 0)])
def test_multi_query_validates_request(query: str, top_k: int) -> None:
    with pytest.raises(ValueError):
        MultiQueryRetriever(QueryAwareRetriever({})).retrieve(query, top_k=top_k)
