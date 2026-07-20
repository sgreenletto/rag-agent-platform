import pytest

from rag_agent_platform.ingestion.chunker import ChunkingConfig, ParentChildChunker


def test_chunker_creates_parent_and_child_chunks() -> None:
    chunker = ParentChildChunker(
        ChunkingConfig(parent_chunk_size=20, child_chunk_size=8, child_overlap=2)
    )

    parents, children = chunker.split(
        document_id="doc-1",
        content="员工申请年假需要提前三个工作日提交申请。",
        source="leave_policy.txt",
        file_type="txt",
    )

    assert parents
    assert children
    assert all(child.document_id == "doc-1" for child in children)
    assert all(child.parent_id in {parent.chunk_id for parent in parents} for child in children)


def test_child_metadata_contains_retrieval_fields() -> None:
    chunker = ParentChildChunker(
        ChunkingConfig(parent_chunk_size=50, child_chunk_size=20, child_overlap=5)
    )

    _, children = chunker.split(
        document_id="doc-1",
        content="采购部门提交采购申请后，财务部门审核预算。",
        source="purchase_process.md",
        file_type="md",
    )

    metadata = children[0].metadata
    assert metadata["document_id"] == children[0].document_id
    assert metadata["parent_id"] == children[0].parent_id
    assert metadata["chunk_id"] == children[0].chunk_id
    assert metadata["source"] == "purchase_process.md"
    assert metadata["file_type"] == "md"
    assert "page" in metadata


def test_child_overlap_repeats_tail_text_in_next_chunk() -> None:
    chunker = ParentChildChunker(
        ChunkingConfig(parent_chunk_size=20, child_chunk_size=10, child_overlap=3)
    )

    _, children = chunker.split(
        document_id="doc-1",
        content="abcdefghijklmnop",
        source="letters.txt",
        file_type="txt",
    )

    assert children[0].content == "abcdefghij"
    assert children[1].content == "hijklmnop"


def test_empty_content_returns_no_chunks() -> None:
    parents, children = ParentChildChunker().split(
        document_id="doc-1",
        content="  \n",
        source="empty.txt",
        file_type="txt",
    )

    assert parents == []
    assert children == []


@pytest.mark.parametrize(
    "config",
    [
        ChunkingConfig(parent_chunk_size=1, child_chunk_size=1, child_overlap=0),
    ],
)
def test_valid_config_is_accepted(config: ChunkingConfig) -> None:
    assert config.parent_chunk_size == 1


@pytest.mark.parametrize(
    "kwargs",
    [
        {"parent_chunk_size": 0},
        {"child_chunk_size": 0},
        {"child_overlap": -1},
        {"child_chunk_size": 10, "child_overlap": 10},
    ],
)
def test_invalid_config_fails(kwargs: dict[str, int]) -> None:
    with pytest.raises(ValueError):
        ChunkingConfig(**kwargs)
