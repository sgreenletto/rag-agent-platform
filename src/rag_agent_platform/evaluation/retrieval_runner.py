"""Execute and compare Retriever modes against an offline question set."""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Any

from rag_agent_platform.evaluation.retrieval_dataset import RetrievalEvaluationCase
from rag_agent_platform.evaluation.retrieval_metrics import calculate_ranking_metrics
from rag_agent_platform.retrieval.base import BaseRetriever


@dataclass(frozen=True, slots=True)
class RetrievalCaseResult:
    case_id: str
    category: str
    retrieved_chunk_ids: tuple[str, ...]
    recall: float
    precision: float
    hit_rate: float
    reciprocal_rank: float
    ndcg: float
    answerability_correct: bool
    latency_ms: float


@dataclass(frozen=True, slots=True)
class RetrievalEvaluationSummary:
    case_count: int
    answerable_case_count: int
    recall: float
    precision: float
    hit_rate: float
    mrr: float
    ndcg: float
    no_answer_accuracy: float
    average_latency_ms: float


@dataclass(frozen=True, slots=True)
class RetrievalModeReport:
    mode: str
    top_k: int
    summary: RetrievalEvaluationSummary
    cases: tuple[RetrievalCaseResult, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RetrievalEvaluationRunner:
    """Evaluate one or more named retrievers with identical cases and top-k."""

    def __init__(self, clock: Callable[[], float] = perf_counter) -> None:
        self._clock = clock

    def evaluate(
        self,
        mode: str,
        retriever: BaseRetriever,
        cases: Sequence[RetrievalEvaluationCase],
        *,
        top_k: int = 5,
    ) -> RetrievalModeReport:
        if not mode.strip():
            raise ValueError("mode must not be empty")
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")
        if not cases:
            raise ValueError("cases must not be empty")

        results: list[RetrievalCaseResult] = []
        for case in cases:
            started = self._clock()
            chunks = retriever.retrieve(
                case.question,
                document_ids=list(case.document_ids) or None,
                top_k=top_k,
            )
            latency_ms = max(0.0, (self._clock() - started) * 1000)
            chunk_ids = [chunk.chunk_id for chunk in chunks]
            metrics = calculate_ranking_metrics(case, chunk_ids, top_k)
            results.append(
                RetrievalCaseResult(
                    case_id=case.case_id,
                    category=case.category,
                    retrieved_chunk_ids=tuple(chunk_ids),
                    recall=metrics.recall,
                    precision=metrics.precision,
                    hit_rate=metrics.hit_rate,
                    reciprocal_rank=metrics.reciprocal_rank,
                    ndcg=metrics.ndcg,
                    answerability_correct=metrics.answerability_correct,
                    latency_ms=latency_ms,
                )
            )
        return RetrievalModeReport(mode, top_k, self._summarize(cases, results), tuple(results))

    def compare(
        self,
        retrievers: Mapping[str, BaseRetriever],
        cases: Sequence[RetrievalEvaluationCase],
        *,
        top_k: int = 5,
    ) -> dict[str, RetrievalModeReport]:
        if not retrievers:
            raise ValueError("retrievers must not be empty")
        return {
            mode: self.evaluate(mode, retriever, cases, top_k=top_k)
            for mode, retriever in retrievers.items()
        }

    @staticmethod
    def _summarize(
        cases: Sequence[RetrievalEvaluationCase],
        results: Sequence[RetrievalCaseResult],
    ) -> RetrievalEvaluationSummary:
        answerable_ids = {case.case_id for case in cases if case.answerable}
        unanswerable_ids = {case.case_id for case in cases if not case.answerable}
        answerable_results = [result for result in results if result.case_id in answerable_ids]
        unanswerable_results = [result for result in results if result.case_id in unanswerable_ids]
        count = len(answerable_results)

        def average(field: str) -> float:
            return sum(float(getattr(result, field)) for result in answerable_results) / count

        return RetrievalEvaluationSummary(
            case_count=len(results),
            answerable_case_count=count,
            recall=average("recall") if count else 0.0,
            precision=average("precision") if count else 0.0,
            hit_rate=average("hit_rate") if count else 0.0,
            mrr=average("reciprocal_rank") if count else 0.0,
            ndcg=average("ndcg") if count else 0.0,
            no_answer_accuracy=(
                sum(result.answerability_correct for result in unanswerable_results)
                / len(unanswerable_results)
                if unanswerable_results
                else 0.0
            ),
            average_latency_ms=sum(result.latency_ms for result in results) / len(results),
        )
