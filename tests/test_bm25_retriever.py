import pytest

from rag_agent_platform.models import ChildChunk
from rag_agent_platform.retrieval.bm25 import BM25Retriever


class MutableCorpus:
    def __init__(self, chunks: list[ChildChunk]) -> None:
        self.chunks = chunks

    def list_chunks(self, document_ids: list[str] | None = None) -> list[ChildChunk]:
        return list(self.chunks)


def chunk(chunk_id: str, document_id: str, content: str) -> ChildChunk:
    return ChildChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        parent_id=f"parent-{chunk_id}",
        content=content,
        metadata={"filename": f"{document_id}.txt"},
    )


@pytest.fixture
def corpus() -> MutableCorpus:
    return MutableCorpus(
        [
            chunk("leave", "hr", "员工每年享有十天带薪年假，休假需要提前申请。"),
            chunk("expense", "finance", "差旅费用报销需要提交发票和审批单。"),
            chunk("security", "it", "Password security requires multi factor authentication."),
        ]
    )


def test_bm25_retrieves_chinese_keyword_and_preserves_metadata(corpus: MutableCorpus) -> None:
    result = BM25Retriever(corpus).retrieve("年假申请", top_k=2)

    assert [item.chunk_id for item in result] == ["leave"]
    assert result[0].retrieval_method == "bm25"
    assert result[0].source == "hr.txt"
    assert "bm25_score" in result[0].metadata


def test_bm25_supports_latin_text_and_document_filter(corpus: MutableCorpus) -> None:
    result = BM25Retriever(corpus).retrieve("PASSWORD SECURITY", document_ids=["it"])

    assert [item.document_id for item in result] == ["it"]


def test_bm25_returns_empty_for_no_keyword_match(corpus: MutableCorpus) -> None:
    assert BM25Retriever(corpus).retrieve("量子物理") == []


def test_bm25_keeps_lexical_matches_even_when_raw_score_is_not_positive() -> None:
    corpus = MutableCorpus(
        [
            chunk("first", "doc-1", "共同词 第一份"),
            chunk("second", "doc-2", "共同词 第二份"),
        ]
    )

    result = BM25Retriever(corpus).retrieve("共同词")

    assert {item.chunk_id for item in result} == {"first", "second"}


def test_bm25_refreshes_after_corpus_change(corpus: MutableCorpus) -> None:
    retriever = BM25Retriever(corpus)
    corpus.chunks.append(chunk("benefit", "hr", "公司提供补充医疗保险福利。"))

    assert retriever.retrieve("医疗保险") == []
    retriever.refresh()
    assert [item.chunk_id for item in retriever.retrieve("医疗保险")] == ["benefit"]


def test_bm25_rejects_invalid_threshold(corpus: MutableCorpus) -> None:
    with pytest.raises(ValueError, match="score_threshold"):
        BM25Retriever(corpus, score_threshold=1.1)


def test_bm25_empty_document_scope_returns_no_results(corpus: MutableCorpus) -> None:
    assert BM25Retriever(corpus).retrieve("年假", document_ids=[]) == []


def test_bm25_refresh_deduplicates_duplicate_chunk_ids() -> None:
    duplicate = chunk("same", "doc", "年假制度")
    retriever = BM25Retriever(MutableCorpus([duplicate, duplicate]))

    assert len(retriever.retrieve("年假")) == 1
