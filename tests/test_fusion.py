import pytest

from rag_agent_platform.models import RetrievedChunk
from rag_agent_platform.retrieval.fusion import reciprocal_rank_fusion


def result(chunk_id: str, score: float, method: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        content=f"content-{chunk_id}",
        normalized_score=score,
        source="source.txt",
        document_id="doc",
        retrieval_method=method,
    )


def test_rrf_promotes_chunk_found_by_both_retrievers() -> None:
    fused = reciprocal_rank_fusion(
        {
            "dense": [result("dense-only", 0.9, "dense"), result("both", 0.8, "dense")],
            "sparse": [result("both", 1.0, "bm25"), result("sparse-only", 0.7, "bm25")],
        },
        rrf_k=10,
    )

    assert fused[0].chunk_id == "both"
    assert fused[0].normalized_score == 1.0
    assert fused[0].retrieval_method == "hybrid_rrf"
    assert set(fused[0].metadata["retriever_results"]) == {"dense", "sparse"}


def test_rrf_weight_changes_ranking() -> None:
    dense = result("dense", 1.0, "dense")
    sparse = result("sparse", 1.0, "bm25")

    fused = reciprocal_rank_fusion(
        {"dense": [dense], "sparse": [sparse]},
        weights={"dense": 2.0, "sparse": 1.0},
    )

    assert [chunk.chunk_id for chunk in fused] == ["dense", "sparse"]


def test_rrf_deduplicates_repeated_chunk_within_one_list() -> None:
    duplicate = result("same", 0.8, "dense")

    fused = reciprocal_rank_fusion({"dense": [duplicate, duplicate]})

    assert len(fused) == 1
    assert fused[0].metadata["retriever_results"]["dense"]["rank"] == 1


@pytest.mark.parametrize("rrf_k", [0, -1])
def test_rrf_rejects_invalid_constant(rrf_k: int) -> None:
    with pytest.raises(ValueError, match="rrf_k"):
        reciprocal_rank_fusion({}, rrf_k=rrf_k)


@pytest.mark.parametrize("weight", [float("nan"), float("inf"), 0.0])
def test_rrf_rejects_non_finite_or_zero_weight(weight: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        reciprocal_rank_fusion(
            {"dense": [result("chunk", 1.0, "dense")]}, weights={"dense": weight}
        )


def test_nested_rrf_preserves_previous_stage_details() -> None:
    hybrid = reciprocal_rank_fusion(
        {"dense": [result("chunk", 1.0, "dense")], "sparse": [result("chunk", 1.0, "bm25")]}
    )

    multi_query = reciprocal_rank_fusion({"query_0": hybrid})

    assert set(multi_query[0].metadata["retrieval_stages"]["hybrid_rrf"]) == {
        "dense",
        "sparse",
    }
    assert set(multi_query[0].metadata["retriever_results"]) == {"query_0"}
