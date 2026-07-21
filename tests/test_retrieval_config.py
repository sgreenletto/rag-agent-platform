import pytest

from rag_agent_platform.config import Settings
from rag_agent_platform.retrieval import RetrievalParameters


def test_retrieval_parameters_read_shared_settings() -> None:
    settings = Settings(
        _env_file=None,
        retrieval_top_k=7,
        dense_candidate_k=21,
        bm25_candidate_k=14,
        rerank_top_k=6,
        retrieval_score_threshold=0.35,
    )

    parameters = RetrievalParameters.from_settings(settings)

    assert parameters == RetrievalParameters(7, 21, 14, 6, 0.35)


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"top_k": 0}, "top_k"),
        ({"dense_candidate_k": 4}, "dense_candidate_k"),
        ({"bm25_candidate_k": 4}, "bm25_candidate_k"),
        ({"rerank_top_k": 0}, "rerank_top_k"),
        ({"score_threshold": -0.1}, "score_threshold"),
        ({"score_threshold": 1.1}, "score_threshold"),
    ],
)
def test_retrieval_parameters_reject_invalid_values(
    overrides: dict[str, int | float], message: str
) -> None:
    values: dict[str, int | float] = {
        "top_k": 5,
        "dense_candidate_k": 20,
        "bm25_candidate_k": 20,
        "rerank_top_k": 5,
        "score_threshold": 0.0,
    }
    values.update(overrides)

    with pytest.raises(ValueError, match=message):
        RetrievalParameters(**values)  # type: ignore[arg-type]
