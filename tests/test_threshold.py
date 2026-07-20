import pytest

from rag_agent_platform.models import RetrievedChunk
from rag_agent_platform.retrieval.base import BaseRetriever
from rag_agent_platform.retrieval.threshold import RelevanceThreshold, ThresholdRetriever


def result(chunk_id: str, score: float, document_id: str = "doc") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        content=f"content-{chunk_id}",
        normalized_score=score,
        source="source.txt",
        document_id=document_id,
        retrieval_method="reranked",
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


def test_threshold_filters_chunks_below_individual_cutoff() -> None:
    policy = RelevanceThreshold(chunk_score_threshold=0.5)

    accepted = policy.apply([result("low", 0.4), result("high", 0.8)])

    assert [chunk.chunk_id for chunk in accepted] == ["high"]


def test_threshold_returns_no_answer_when_top_score_is_too_low() -> None:
    policy = RelevanceThreshold(min_top_score=0.8)

    assert policy.apply([result("weak", 0.79)]) == []


def test_threshold_returns_no_answer_when_too_few_results_remain() -> None:
    policy = RelevanceThreshold(chunk_score_threshold=0.5, min_results=2)

    assert policy.apply([result("strong", 0.9), result("weak", 0.4)]) == []


def test_threshold_can_require_separation_from_second_result() -> None:
    policy = RelevanceThreshold(min_score_gap=0.2)

    assert policy.apply([result("a", 0.9), result("b", 0.8)]) == []
    assert [chunk.chunk_id for chunk in policy.apply([result("a", 0.9), result("b", 0.6)])] == [
        "a",
        "b",
    ]


def test_threshold_sorts_accepted_results() -> None:
    accepted = RelevanceThreshold().apply([result("low", 0.3), result("high", 0.9)])

    assert [chunk.chunk_id for chunk in accepted] == ["high", "low"]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"chunk_score_threshold": -0.1}, "chunk_score_threshold"),
        ({"min_top_score": 1.1}, "min_top_score"),
        ({"min_score_gap": 1.1}, "min_score_gap"),
        ({"min_results": 0}, "min_results"),
    ],
)
def test_threshold_rejects_invalid_configuration(
    kwargs: dict[str, float | int], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        RelevanceThreshold(**kwargs)  # type: ignore[arg-type]


def test_threshold_retriever_expands_candidates_for_policy() -> None:
    base = RecordingRetriever([result("a", 0.9), result("b", 0.7)])
    retriever = ThresholdRetriever(base, RelevanceThreshold(min_results=2), candidate_multiplier=3)

    accepted = retriever.retrieve("  query  ", document_ids=["doc"], top_k=1)

    assert base.call == ("query", ["doc"], 3)
    assert [chunk.chunk_id for chunk in accepted] == ["a"]


def test_threshold_retriever_rejects_invalid_candidate_multiplier() -> None:
    with pytest.raises(ValueError, match="candidate_multiplier"):
        ThresholdRetriever(RecordingRetriever([]), RelevanceThreshold(), candidate_multiplier=0)


def test_duplicate_candidates_cannot_satisfy_minimum_result_count() -> None:
    duplicate = result("same", 0.9)
    retriever = ThresholdRetriever(
        RecordingRetriever([duplicate, duplicate]), RelevanceThreshold(min_results=2)
    )

    assert retriever.retrieve("query") == []
