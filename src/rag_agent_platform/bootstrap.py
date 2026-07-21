"""Central service container for real and explicit Mock application modes."""

from dataclasses import dataclass
from pathlib import Path

from rag_agent_platform.agent import AgentService, LangGraphAgentService, MockAgentService
from rag_agent_platform.agent.router import BoundedQueryRewriter, StructuredQueryAnalyzer
from rag_agent_platform.config import Settings, settings
from rag_agent_platform.embeddings import build_embedding_model
from rag_agent_platform.evaluation import GroundedAnswerEvaluator
from rag_agent_platform.generation import GroundedAnswerGenerator
from rag_agent_platform.graph import GraphRetriever, MockTripletExtractor, NetworkXGraphService
from rag_agent_platform.ingestion import (
    CoordinatedIngestionPipeline,
    IngestionPipeline,
    MockIngestionPipeline,
    RealIngestionPipeline,
)
from rag_agent_platform.llm import build_chat_model
from rag_agent_platform.retrieval import (
    AdvancedRetriever,
    BM25Retriever,
    CompressionRetriever,
    DenseRetriever,
    HybridRetriever,
    MockRetriever,
    MultiQueryRetriever,
    NaiveRetriever,
    ParentContextRetriever,
    RerankingRetriever,
    SentenceContextCompressor,
    TokenOverlapReranker,
)
from rag_agent_platform.storage import (
    ChromaDenseSearchBackend,
    ChromaVectorStore,
    DocumentRepository,
    MockDocumentRepository,
    RepositoryChunkCorpus,
    build_document_repository,
)


@dataclass(slots=True)
class ServiceContainer:
    """Stable object graph cached by the Streamlit process."""

    settings: Settings
    ingestion: IngestionPipeline
    repository: DocumentRepository
    agent: AgentService
    app_mode: str
    llm_configured: bool


def build_service_container(config: Settings | None = None) -> ServiceContainer:
    """Build all long-lived services once, with no silent Mock fallback."""
    active_settings = config or settings
    app_mode = active_settings.app_mode.strip().lower()
    if app_mode == "mock":
        return _build_mock_container(active_settings)
    if app_mode != "real":
        raise ValueError("APP_MODE must be either 'real' or 'mock'")
    return _build_real_container(active_settings)


def _build_real_container(config: Settings) -> ServiceContainer:
    repository = build_document_repository(config)
    embedding_model = build_embedding_model(config)
    vector_store = ChromaVectorStore(
        persist_directory=config.chroma_persist_directory,
        collection_name=config.chroma_collection_name,
        embedding_model=embedding_model,
    )
    real_ingestion = RealIngestionPipeline(repository=repository, vector_store=vector_store)

    corpus = RepositoryChunkCorpus(repository)
    sparse = BM25Retriever(corpus)
    dense = DenseRetriever(
        ChromaDenseSearchBackend(vector_store),
        score_kind="distance",
    )
    naive = NaiveRetriever(ParentContextRetriever(dense, repository))

    hybrid = HybridRetriever(dense, sparse)
    multi_query = MultiQueryRetriever(hybrid)
    reranked = RerankingRetriever(multi_query, TokenOverlapReranker())
    parent_context = ParentContextRetriever(reranked, repository)
    advanced = AdvancedRetriever(
        CompressionRetriever(
            parent_context,
            SentenceContextCompressor(),
            max_chars=6000,
        )
    )

    graph_service = NetworkXGraphService(
        MockTripletExtractor(),
        persist_dir=Path(config.graph_persist_directory),
        document_sources={
            document.document_id: document.filename for document in repository.list_documents()
        },
    )
    graph_service.build(repository.list_child_chunks())
    graph_retriever = GraphRetriever(graph_service)

    ingestion = CoordinatedIngestionPipeline(
        real_ingestion,
        repository=repository,
        sparse_retriever=sparse,
        graph_service=graph_service,
    )
    chat_model = build_chat_model(config)
    generator = GroundedAnswerGenerator(chat_model)
    evaluator = GroundedAnswerEvaluator(chat_model)
    agent = LangGraphAgentService(
        naive_retriever=naive,
        advanced_retriever=advanced,
        graph_retriever=graph_retriever,
        generator=generator,
        evaluator=evaluator,
        analyzer=StructuredQueryAnalyzer(chat_model),
        rewriter=BoundedQueryRewriter(chat_model),
        top_k=config.retrieval_top_k,
        max_retries=config.agent_max_retries,
    )
    return ServiceContainer(
        settings=config,
        ingestion=ingestion,
        repository=repository,
        agent=agent,
        app_mode="real",
        llm_configured=chat_model is not None,
    )


def _build_mock_container(config: Settings) -> ServiceContainer:
    repository = MockDocumentRepository()
    ingestion = MockIngestionPipeline(repository)
    agent = MockAgentService(
        naive_retriever=MockRetriever(retrieval_method="naive"),
        advanced_retriever=MockRetriever(retrieval_method="advanced"),
        graph_retriever=MockRetriever(retrieval_method="graph"),
    )
    return ServiceContainer(
        settings=config,
        ingestion=ingestion,
        repository=repository,
        agent=agent,
        app_mode="mock",
        llm_configured=False,
    )
