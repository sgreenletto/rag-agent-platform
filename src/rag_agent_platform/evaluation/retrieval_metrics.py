"""Standard ranking metrics for offline retrieval evaluation."""

from dataclasses import dataclass
from math import log2

from rag_agent_platform.evaluation.retrieval_dataset import RetrievalEvaluationCase


@dataclass(frozen=True, slots=True)
class RankingMetrics:
    recall: float
    precision: float
    hit_rate: float
    reciprocal_rank: float
    ndcg: float
    answerability_correct: bool


def calculate_ranking_metrics(
    case: RetrievalEvaluationCase,
    retrieved_chunk_ids: list[str],
    k: int,
) -> RankingMetrics:
    """Calculate binary-relevance metrics at k for one evaluation case."""
    if k <= 0:
        raise ValueError("k must be greater than 0")
    ranked_ids = _deduplicate(retrieved_chunk_ids[:k])
    relevant = set(case.relevant_chunk_ids)
    hits = [chunk_id in relevant for chunk_id in ranked_ids]
    hit_count = sum(hits)
    answerability_correct = case.answerable == bool(ranked_ids)

    if not case.answerable:
        return RankingMetrics(0.0, 0.0, 0.0, 0.0, 0.0, answerability_correct)

    recall = hit_count / len(relevant)
    precision = hit_count / k
    hit_rate = float(hit_count > 0)
    reciprocal_rank = next((1.0 / rank for rank, hit in enumerate(hits, start=1) if hit), 0.0)
    dcg = sum(1.0 / log2(rank + 1) for rank, hit in enumerate(hits, start=1) if hit)
    ideal_hits = min(len(relevant), k)
    ideal_dcg = sum(1.0 / log2(rank + 1) for rank in range(1, ideal_hits + 1))
    ndcg = dcg / ideal_dcg if ideal_dcg else 0.0
    return RankingMetrics(recall, precision, hit_rate, reciprocal_rank, ndcg, answerability_correct)


def _deduplicate(chunk_ids: list[str]) -> list[str]:
    return list(dict.fromkeys(chunk_ids))
