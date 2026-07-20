import pytest

from rag_agent_platform.models import RetrievedChunk
from rag_agent_platform.retrieval.base import BaseRetriever
from rag_agent_platform.retrieval.compression import (
    CompressionRetriever,
    ContextCompressor,
    SentenceContextCompressor,
)


def result(chunk_id: str, content: str, score: float = 1.0) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        content=content,
        normalized_score=score,
        source="policy.txt",
        document_id="doc",
        parent_id="parent",
        page=2,
        retrieval_method="hybrid_rrf",
        metadata={"existing": True},
    )


class RecordingRetriever(BaseRetriever):
    def __init__(self, chunks: list[RetrievedChunk]) -> None:
        self.chunks = chunks
        self.call: tuple[str, list[str] | None, int] | None = None

    def retrieve(
        self,
        query: str,
        document_ids: list[str] | None = None,
        top_k: int = 5,
    ) -> list[RetrievedChunk]:
        self.call = (query, document_ids, top_k)
        return self.chunks[:top_k]


class InjectingCompressor(ContextCompressor):
    def compress(
        self, query: str, chunks: list[RetrievedChunk], max_chars: int
    ) -> list[RetrievedChunk]:
        return [result("injected", "not retrieved")]


def test_sentence_compressor_selects_relevant_sentence_and_preserves_identity() -> None:
    original = result("chunk", "办公地点在上海。员工年假为十天。报销需要发票。")

    compressed = SentenceContextCompressor().compress("年假几天", [original], max_chars=10)

    assert compressed[0].content == "员工年假为十天。"
    assert compressed[0].chunk_id == original.chunk_id
    assert compressed[0].source == original.source
    assert compressed[0].page == original.page
    assert compressed[0].normalized_score == original.normalized_score


def test_sentence_compressor_respects_global_budget() -> None:
    chunks = [result("a", "第一段完整内容。"), result("b", "第二段完整内容。")]

    compressed = SentenceContextCompressor().compress("内容", chunks, max_chars=10)

    assert sum(len(chunk.content) for chunk in compressed) <= 10


def test_sentence_compressor_truncates_relevant_sentence_before_using_irrelevant_short_text() -> (
    None
):
    original = result("chunk", "年假申请需要提前提交审批材料。无关。")

    compressed = SentenceContextCompressor().compress("年假申请", [original], max_chars=6)

    assert compressed[0].content == "年假申请需要"


def test_sentence_compressor_does_not_mutate_original_metadata() -> None:
    original = result("chunk", "年假制度内容很长。其他内容。")

    compressed = SentenceContextCompressor().compress("年假", [original], max_chars=8)

    assert original.metadata == {"existing": True}
    assert compressed[0].metadata["original_content_length"] == len(original.content)
    assert compressed[0].metadata["compressed_content_length"] == len(compressed[0].content)


def test_sentence_compressor_returns_empty_for_empty_input() -> None:
    assert SentenceContextCompressor().compress("query", [], max_chars=10) == []


@pytest.mark.parametrize(("query", "max_chars"), [(" ", 10), ("query", 0)])
def test_sentence_compressor_validates_request(query: str, max_chars: int) -> None:
    with pytest.raises(ValueError):
        SentenceContextCompressor().compress(query, [], max_chars)


def test_compression_retriever_forwards_request() -> None:
    base = RecordingRetriever([result("chunk", "年假制度。")])
    retriever = CompressionRetriever(base, SentenceContextCompressor(), max_chars=20)

    retrieved = retriever.retrieve("  年假  ", document_ids=["doc"], top_k=3)

    assert base.call == ("年假", ["doc"], 3)
    assert [chunk.chunk_id for chunk in retrieved] == ["chunk"]


def test_compression_retriever_rejects_invalid_budget() -> None:
    with pytest.raises(ValueError, match="max_chars"):
        CompressionRetriever(RecordingRetriever([]), SentenceContextCompressor(), max_chars=0)


def test_compression_retriever_rejects_injected_evidence() -> None:
    retriever = CompressionRetriever(
        RecordingRetriever([result("real", "retrieved")]), InjectingCompressor()
    )

    with pytest.raises(ValueError, match="unknown chunk_id"):
        retriever.retrieve("query")
