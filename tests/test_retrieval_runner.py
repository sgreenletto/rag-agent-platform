from collections.abc import Iterator

import pytest

from rag_agent_platform.evaluation.retrieval_dataset import RetrievalEvaluationCase
from rag_agent_platform.evaluation.retrieval_runner import RetrievalEvaluationRunner
from rag_agent_platform.models import RetrievedChunk
from rag_agent_platform.retrieval.base import BaseRetriever


def result(chunk_id: str) -> RetrievedChunk:
    return RetrievedChunk(chunk_id, chunk_id, 1.0, "source.txt", retrieval_method="fixture")


class CaseRetriever(BaseRetriever):
    def __init__(self, answers: dict[str, list[RetrievedChunk]]) -> None:
        self.answers = answers
        self.calls: list[tuple[str, list[str] | None, int]] = []

    def retrieve(
        self, query: str, document_ids: list[str] | None = None, top_k: int = 5
    ) -> list[RetrievedChunk]:
        self.calls.append((query, document_ids, top_k))
        return self.answers.get(query, [])[:top_k]


def clock(values: list[float]) -> Iterator[float]:
    yield from values


def test_runner_aggregates_answerable_metrics_and_no_answer_accuracy() -> None:
    cases = [
        RetrievalEvaluationCase("a", "known", ("right",), ("doc",), document_ids=("selected",)),
        RetrievalEvaluationCase("b", "unknown", (), answerable=False),
    ]
    retriever = CaseRetriever({"known": [result("right")], "unknown": []})
    times = clock([0.0, 0.01, 0.02, 0.04])
    report = RetrievalEvaluationRunner(clock=lambda: next(times)).evaluate(
        "naive", retriever, cases, top_k=2
    )

    assert report.mode == "naive"
    assert report.summary.recall == 1.0
    assert report.summary.precision == 0.5
    assert report.summary.mrr == 1.0
    assert report.summary.ndcg == 1.0
    assert report.summary.no_answer_accuracy == 1.0
    assert report.summary.average_latency_ms == pytest.approx(15.0)
    assert retriever.calls[0] == ("known", ["selected"], 2)


def test_runner_does_not_leak_relevant_documents_into_retrieval_filter() -> None:
    case = RetrievalEvaluationCase("a", "query", ("right",), ("relevant-doc",))
    retriever = CaseRetriever({"query": [result("right")]})

    RetrievalEvaluationRunner().evaluate("mode", retriever, [case])

    assert retriever.calls[0][1] is None


def test_no_answer_accuracy_only_uses_unanswerable_cases() -> None:
    cases = [
        RetrievalEvaluationCase("missed", "known", ("right",)),
        RetrievalEvaluationCase("unknown", "unknown", (), answerable=False),
    ]
    retriever = CaseRetriever({"known": [], "unknown": []})

    report = RetrievalEvaluationRunner().evaluate("mode", retriever, cases)

    assert report.summary.no_answer_accuracy == 1.0


def test_runner_compares_named_modes() -> None:
    cases = [RetrievalEvaluationCase("a", "query", ("right",))]
    runner = RetrievalEvaluationRunner()

    reports = runner.compare(
        {
            "naive": CaseRetriever({"query": [result("wrong")]}),
            "advanced": CaseRetriever({"query": [result("right")]}),
        },
        cases,
        top_k=1,
    )

    assert reports["naive"].summary.hit_rate == 0.0
    assert reports["advanced"].summary.hit_rate == 1.0
    assert reports["advanced"].to_dict()["mode"] == "advanced"


@pytest.mark.parametrize(
    ("mode", "top_k", "cases", "message"),
    [
        (" ", 1, [RetrievalEvaluationCase("a", "q", ("c",))], "mode"),
        ("mode", 0, [RetrievalEvaluationCase("a", "q", ("c",))], "top_k"),
        ("mode", 1, [], "cases"),
    ],
)
def test_runner_rejects_invalid_evaluation_request(
    mode: str, top_k: int, cases: list[RetrievalEvaluationCase], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        RetrievalEvaluationRunner().evaluate(mode, CaseRetriever({}), cases, top_k=top_k)


def test_runner_rejects_empty_mode_comparison() -> None:
    with pytest.raises(ValueError, match="retrievers"):
        RetrievalEvaluationRunner().compare({}, [RetrievalEvaluationCase("a", "q", ("c",))])
