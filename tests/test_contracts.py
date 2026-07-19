from rag_agent_platform.models import ParentChunk


def test_public_model_exports_are_available() -> None:
    chunk = ParentChunk(
        chunk_id="parent-1",
        document_id="doc-1",
        content="父块内容",
    )

    assert chunk.document_id == "doc-1"
