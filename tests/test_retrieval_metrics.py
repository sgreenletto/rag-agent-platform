import pytest

from rag_agent_platform.evaluation.retrieval_dataset import RetrievalEvaluationCase
from rag_agent_platform.evaluation.retrieval_metrics import calculate_ranking_metrics


def case(*relevant: str, answerable: bool = True) -> RetrievalEvaluationCase:
    return RetrievalEvaluationCase("case", "question", relevant, answerable=answerable)


def test_metrics_calculate_binary_ranking_quality_at_k() -> None:
    metrics = calculate_ranking_metrics(case("a", "b"), ["x", "a", "b"], k=3)

    assert metrics.recall == 1.0
    assert metrics.precision == pytest.approx(2 / 3)
    assert metrics.hit_rate == 1.0
    assert metrics.reciprocal_rank == 0.5
    assert 0.0 < metrics.ndcg < 1.0


def test_metrics_deduplicate_retrieved_chunk_ids() -> None:
    metrics = calculate_ranking_metrics(case("a", "b"), ["a", "a", "b"], k=2)

    assert metrics.recall == 0.5
    assert metrics.precision == 0.5


def test_metrics_for_miss_are_zero() -> None:
    metrics = calculate_ranking_metrics(case("relevant"), ["other"], k=1)

    assert metrics.recall == 0.0
    assert metrics.hit_rate == 0.0
    assert metrics.reciprocal_rank == 0.0
    assert metrics.ndcg == 0.0


def test_unanswerable_case_only_scores_answerability() -> None:
    metrics = calculate_ranking_metrics(case(answerable=False), [], k=5)

    assert metrics.answerability_correct is True
    assert metrics.recall == 0.0


def test_unanswerable_case_with_results_is_incorrect() -> None:
    assert (
        calculate_ranking_metrics(case(answerable=False), ["noise"], 5).answerability_correct
        is False
    )


def test_metrics_reject_invalid_k() -> None:
    with pytest.raises(ValueError, match="k"):
        calculate_ranking_metrics(case("a"), [], 0)
