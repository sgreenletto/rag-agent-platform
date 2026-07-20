from rag_agent_platform.models import ChildChunk, RetrievedChunk
from rag_agent_platform.retrieval import (
    BM25Retriever,
    CompressionRetriever,
    DenseRetriever,
    DenseSearchHit,
    HybridRetriever,
    MultiQueryRetriever,
    RelevanceThreshold,
    RerankingRetriever,
    SentenceContextCompressor,
    ThresholdRetriever,
    TokenOverlapReranker,
)


class InMemoryChunkCorpus:
    def __init__(self, chunks: list[ChildChunk]) -> None:
        self.chunks = chunks

    def list_chunks(self, document_ids: list[str] | None = None) -> list[ChildChunk]:
        allowed = None if document_ids is None else set(document_ids)
        return [chunk for chunk in self.chunks if allowed is None or chunk.document_id in allowed]


class StaticTransformer:
    def transform(self, query: str) -> list[str]:
        return ["休假 天数"] if "年假" in query else []


class FakeDenseSearchBackend:
    def __init__(self, chunks: list[ChildChunk], fail: bool = False) -> None:
        self.chunks = chunks
        self.fail = fail

    def search(
        self, query: str, document_ids: list[str] | None, limit: int
    ) -> list[DenseSearchHit]:
        if self.fail:
            raise ConnectionError("dense unavailable")
        if not any(term in query for term in ("年假", "休假", "报销", "费用")):
            return []
        allowed = None if document_ids is None else set(document_ids)
        hits = []
        for chunk in self.chunks:
            if allowed is not None and chunk.document_id not in allowed:
                continue
            score = (
                0.9 if chunk.chunk_id == "leave" and ("年假" in query or "休假" in query) else 0.1
            )
            hits.append(
                DenseSearchHit(
                    chunk=chunk,
                    score=score,
                    source=str(chunk.metadata["filename"]),
                )
            )
        return sorted(hits, key=lambda hit: hit.score, reverse=True)[:limit]


def build_chunks() -> list[ChildChunk]:
    return [
        ChildChunk(
            "leave",
            "hr",
            "parent-hr",
            "员工每年享有十天带薪年假。休假需要提前申请。",
            page=1,
            metadata={"filename": "hr.txt"},
        ),
        ChildChunk(
            "expense",
            "finance",
            "parent-finance",
            "差旅费用报销需要提交发票。",
            page=2,
            metadata={"filename": "finance.txt"},
        ),
    ]


def build_pipeline(chunks: list[ChildChunk], *, dense_fails: bool = False) -> ThresholdRetriever:
    dense = DenseRetriever(FakeDenseSearchBackend(chunks, fail=dense_fails))
    sparse = BM25Retriever(InMemoryChunkCorpus(chunks))
    hybrid = HybridRetriever(dense, sparse)
    multi_query = MultiQueryRetriever(hybrid, StaticTransformer())
    reranked = RerankingRetriever(multi_query, TokenOverlapReranker())
    compressed = CompressionRetriever(reranked, SentenceContextCompressor(), max_chars=30)
    return ThresholdRetriever(compressed, RelevanceThreshold(min_top_score=0.1))


def test_complete_advanced_retrieval_pipeline_preserves_contract_and_provenance() -> None:
    chunks = build_chunks()
    original_state = [(chunk.content, dict(chunk.metadata)) for chunk in chunks]

    results = build_pipeline(chunks).retrieve("员工年假有几天？", document_ids=["hr"], top_k=2)

    assert results
    assert all(isinstance(chunk, RetrievedChunk) for chunk in results)
    assert len(results) <= 2
    assert len({chunk.chunk_id for chunk in results}) == len(results)
    assert all(chunk.document_id == "hr" for chunk in results)
    assert all(0.0 <= chunk.normalized_score <= 1.0 for chunk in results)
    assert results == sorted(results, key=lambda chunk: (-chunk.normalized_score, chunk.chunk_id))
    assert results[0].source == "hr.txt"
    assert results[0].page == 1
    assert "retriever_results" in results[0].metadata
    assert set(results[0].metadata["retrieval_stages"]["hybrid_rrf"]) == {
        "dense",
        "sparse",
    }
    assert "rerank_score" in results[0].metadata
    assert "compression_method" in results[0].metadata
    assert [(chunk.content, chunk.metadata) for chunk in chunks] == original_state


def test_complete_pipeline_enforces_empty_scope_and_no_answer() -> None:
    pipeline = build_pipeline(build_chunks())

    assert pipeline.retrieve("员工年假有几天？", document_ids=[]) == []
    assert pipeline.retrieve("火星旅行补贴") == []


def test_complete_pipeline_degrades_when_dense_backend_fails() -> None:
    results = build_pipeline(build_chunks(), dense_fails=True).retrieve(
        "员工年假有几天？", document_ids=["hr"]
    )

    assert results[0].chunk_id == "leave"
    assert "dense" in results[0].metadata["retrieval_warnings"]
