import pytest

from rag_agent_platform.models.schemas import RetrievedChunk


def build_chunk(**overrides: object) -> RetrievedChunk:
    values: dict[str, object] = {
        "chunk_id": "chunk-1",
        "content": "有效内容",
        "normalized_score": 0.8,
        "source": "source.txt",
    }
    values.update(overrides)
    return RetrievedChunk(**values)  # type: ignore[arg-type]


def test_retrieved_chunk_can_be_created() -> None:
    chunk = build_chunk()

    assert chunk.normalized_score == 0.8
    assert chunk.content == "有效内容"


@pytest.mark.parametrize("score", [-0.01, 1.01])
def test_retrieved_chunk_rejects_out_of_range_score(score: float) -> None:
    with pytest.raises(ValueError, match="normalized_score"):
        build_chunk(normalized_score=score)


def test_retrieved_chunk_rejects_empty_content() -> None:
    with pytest.raises(ValueError, match="content"):
        build_chunk(content="   ")


def test_retrieved_chunk_rejects_empty_source() -> None:
    with pytest.raises(ValueError, match="source"):
        build_chunk(source="")
