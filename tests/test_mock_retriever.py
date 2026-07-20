import pytest

from rag_agent_platform.models.schemas import RetrievedChunk
from rag_agent_platform.retrieval.mock import MockRetriever


@pytest.fixture
def retriever() -> MockRetriever:
    return MockRetriever(retrieval_method="naive")


def test_returns_retrieved_chunk_list(retriever: MockRetriever) -> None:
    results = retriever.retrieve("请假制度")

    assert isinstance(results, list)
    assert results
    assert all(isinstance(chunk, RetrievedChunk) for chunk in results)
    assert results == sorted(results, key=lambda chunk: chunk.normalized_score, reverse=True)


def test_document_ids_filter_is_applied(retriever: MockRetriever) -> None:
    results = retriever.retrieve("请假制度", document_ids=["mock-document-2"])

    assert results
    assert all(chunk.document_id == "mock-document-2" for chunk in results)


def test_top_k_is_applied(retriever: MockRetriever) -> None:
    results = retriever.retrieve("请假制度", top_k=2)

    assert len(results) == 2


def test_empty_query_fails(retriever: MockRetriever) -> None:
    with pytest.raises(ValueError, match="query"):
        retriever.retrieve("   ")


@pytest.mark.parametrize("top_k", [0, -1])
def test_invalid_top_k_fails(retriever: MockRetriever, top_k: int) -> None:
    with pytest.raises(ValueError, match="top_k"):
        retriever.retrieve("问题", top_k=top_k)
