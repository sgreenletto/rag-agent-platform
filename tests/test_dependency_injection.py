import inspect
from typing import get_type_hints

from rag_agent_platform.agent.base import AgentService
from rag_agent_platform.config import Settings
from rag_agent_platform.evaluation.base import AnswerEvaluator
from rag_agent_platform.generation import GroundedFallbackSynthesizer
from rag_agent_platform.generation.base import AnswerGenerator, FallbackSynthesizer
from rag_agent_platform.ingestion.base import IngestionPipeline
from rag_agent_platform.ingestion.pipeline import RealIngestionPipeline
from rag_agent_platform.retrieval.base import BaseRetriever
from rag_agent_platform.storage.base import DocumentRepository


def test_real_ingestion_requires_repository_injection() -> None:
    parameter = inspect.signature(RealIngestionPipeline).parameters["repository"]

    assert parameter.default is inspect.Parameter.empty


def test_application_services_exposes_all_injected_boundaries() -> None:
    from rag_agent_platform.bootstrap import ApplicationServices, build_application_services

    services = build_application_services(Settings(_env_file=None, app_mode="mock"))

    assert isinstance(services, ApplicationServices)
    assert isinstance(services.ingestion, IngestionPipeline)
    assert isinstance(services.repository, DocumentRepository)
    assert isinstance(services.naive_retriever, BaseRetriever)
    assert isinstance(services.advanced_retriever, BaseRetriever)
    assert isinstance(services.graph_retriever, BaseRetriever)
    assert isinstance(services.generator, AnswerGenerator)
    assert isinstance(services.evaluator, AnswerEvaluator)
    assert isinstance(services.agent, AgentService)


def test_streamlit_build_services_returns_application_services() -> None:
    from rag_agent_platform.bootstrap import ApplicationServices
    from rag_agent_platform.ui.app import build_services

    assert get_type_hints(build_services)["return"] is ApplicationServices


def test_grounded_fallback_satisfies_injected_protocol() -> None:
    assert isinstance(GroundedFallbackSynthesizer(), FallbackSynthesizer)
