import math

import pytest

from rag_agent_platform.models import ChildChunk, RetrievedChunk
from rag_agent_platform.retrieval import ChunkCorpus
from rag_agent_platform.retrieval.validation import (
    finalize_results,
    min_max_normalize,
    validate_retrieval_request,
)


class InMemoryCorpus:
    def __init__(self, chunks: list[ChildChunk]) -> None:
        self._chunks = chunks

    def list_chunks(self, document_ids: list[str] | None = None) -> list[ChildChunk]:
        allowed = set(document_ids) if document_ids else None
        return [chunk for chunk in self._chunks if allowed is None or chunk.document_id in allowed]


def make_result(chunk_id: str, document_id: str, score: float) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        content=f"content-{chunk_id}",
        normalized_score=score,
        source="fixture.txt",
        document_id=document_id,
        retrieval_method="fixture",
    )


def test_chunk_corpus_is_a_storage_agnostic_runtime_contract() -> None:
    corpus = InMemoryCorpus([ChildChunk("chunk-1", "doc-1", "parent-1", "测试语料")])

    assert isinstance(corpus, ChunkCorpus)
    assert corpus.list_chunks(["doc-1"])[0].chunk_id == "chunk-1"


@pytest.mark.parametrize(("scores", "expected"), [([], []), ([2.0], [1.0]), ([0.0], [1.0])])
def test_min_max_normalize_edge_cases(scores: list[float], expected: list[float]) -> None:
    assert min_max_normalize(scores) == expected


def test_min_max_normalize_preserves_order_and_bounds() -> None:
    assert min_max_normalize([-2.0, 0.0, 6.0]) == [0.0, 0.25, 1.0]


@pytest.mark.parametrize("score", [math.inf, -math.inf, math.nan])
def test_min_max_normalize_rejects_non_finite_scores(score: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        min_max_normalize([score])


def test_finalize_results_filters_orders_and_limits_deterministically() -> None:
    chunks = [
        make_result("chunk-b", "doc-1", 0.8),
        make_result("chunk-a", "doc-1", 0.8),
        make_result("chunk-c", "doc-2", 0.9),
    ]

    results = finalize_results(chunks, document_ids=["doc-1"], top_k=1)

    assert [chunk.chunk_id for chunk in results] == ["chunk-a"]


def test_finalize_results_treats_empty_document_scope_as_no_documents() -> None:
    assert finalize_results([make_result("chunk", "doc", 1.0)], [], 5) == []


def test_finalize_results_deduplicates_by_chunk_id_using_highest_score() -> None:
    results = finalize_results(
        [make_result("same", "doc", 0.4), make_result("same", "doc", 0.9)], None, 5
    )

    assert len(results) == 1
    assert results[0].normalized_score == 0.9


def test_validate_retrieval_request_returns_stripped_query() -> None:
    assert validate_retrieval_request("  年假制度  ", 5) == "年假制度"
