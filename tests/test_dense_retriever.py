import pytest

from rag_agent_platform.models import ChildChunk
from rag_agent_platform.retrieval.dense import DenseRetriever, DenseSearchHit


def hit(chunk_id: str, document_id: str, score: float) -> DenseSearchHit:
    return DenseSearchHit(
        chunk=ChildChunk(
            chunk_id=chunk_id,
            document_id=document_id,
            parent_id=f"parent-{chunk_id}",
            content=f"content-{chunk_id}",
        ),
        score=score,
        source=f"{document_id}.txt",
    )


class FakeDenseBackend:
    def __init__(self, hits: list[DenseSearchHit]) -> None:
        self.hits = hits
        self.last_call: tuple[str, list[str] | None, int] | None = None

    def search(
        self, query: str, document_ids: list[str] | None, limit: int
    ) -> list[DenseSearchHit]:
        self.last_call = (query, document_ids, limit)
        allowed = set(document_ids) if document_ids else None
        return [item for item in self.hits if allowed is None or item.chunk.document_id in allowed][
            :limit
        ]


def test_dense_similarity_normalizes_and_forwards_backend_filters() -> None:
    backend = FakeDenseBackend([hit("a", "doc-1", 0.2), hit("b", "doc-1", 0.9)])
    retriever = DenseRetriever(backend, candidate_multiplier=3)

    result = retriever.retrieve("  年假  ", document_ids=["doc-1"], top_k=1)

    assert [item.chunk_id for item in result] == ["b"]
    assert result[0].normalized_score == 1.0
    assert result[0].metadata["dense_score"] == 0.9
    assert result[0].metadata["raw_score"] == 0.9
    assert result[0].metadata["score_type"] == "similarity"
    assert result[0].metadata["normalized_score"] == 1.0
    assert backend.last_call == ("年假", ["doc-1"], 3)


def test_dense_distance_treats_smaller_score_as_more_relevant() -> None:
    backend = FakeDenseBackend([hit("far", "doc", 1.5), hit("near", "doc", 0.1)])

    result = DenseRetriever(backend, score_kind="distance").retrieve("query")

    assert [item.chunk_id for item in result] == ["near", "far"]
    assert [item.normalized_score for item in result] == [1.0, 0.0]


def test_dense_applies_normalized_no_answer_threshold() -> None:
    backend = FakeDenseBackend([hit("low", "doc", 0.1), hit("high", "doc", 0.2)])

    result = DenseRetriever(backend, score_threshold=0.5).retrieve("query")

    assert [item.chunk_id for item in result] == ["high"]


@pytest.mark.parametrize("score_kind", ["cosine", "euclidean"])
def test_dense_rejects_unknown_score_kind(score_kind: str) -> None:
    with pytest.raises(ValueError, match="score_kind"):
        DenseRetriever(FakeDenseBackend([]), score_kind=score_kind)  # type: ignore[arg-type]


def test_dense_defensively_caps_and_deduplicates_backend_results() -> None:
    duplicate = hit("same", "doc", 0.9)
    backend = FakeDenseBackend([duplicate, duplicate, hit("extra", "doc", 0.1)])

    results = DenseRetriever(backend, candidate_multiplier=2).retrieve("query", top_k=1)

    assert [item.chunk_id for item in results] == ["same"]


def test_dense_empty_document_scope_cannot_leak_backend_results() -> None:
    backend = FakeDenseBackend([hit("chunk", "doc", 0.9)])

    assert DenseRetriever(backend).retrieve("query", document_ids=[]) == []


def test_single_distance_candidate_uses_absolute_distance_confidence() -> None:
    backend = FakeDenseBackend([hit("only", "doc", 0.7)])

    result = DenseRetriever(backend, score_kind="distance", score_threshold=0.5).retrieve("query")

    assert result[0].normalized_score == pytest.approx(1.0 / 1.7)


def test_dense_explicit_candidate_k_overrides_multiplier() -> None:
    backend = FakeDenseBackend([hit("a", "doc", 0.9)])

    DenseRetriever(backend, candidate_multiplier=9, candidate_k=7).retrieve("query", top_k=5)

    assert backend.last_call == ("query", None, 7)


def test_dense_rejects_candidate_k_smaller_than_requested_top_k() -> None:
    retriever = DenseRetriever(FakeDenseBackend([]), candidate_k=2)

    with pytest.raises(ValueError, match="candidate_k"):
        retriever.retrieve("query", top_k=3)
