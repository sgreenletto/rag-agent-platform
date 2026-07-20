def test_public_modules_import_without_cycles() -> None:
    from rag_agent_platform.models.schemas import RetrievedChunk

    from rag_agent_platform.agent.mock import MockAgentService
    from rag_agent_platform.retrieval.mock import MockRetriever

    assert RetrievedChunk.__module__ == "rag_agent_platform.models.schemas"
    assert MockRetriever.__module__ == "rag_agent_platform.retrieval.mock"
    assert MockAgentService.__module__ == "rag_agent_platform.agent.mock"
