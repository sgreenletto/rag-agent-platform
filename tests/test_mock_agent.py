import pytest

from rag_agent_platform.agent.mock import MockAgentService
from rag_agent_platform.models.schemas import AgentResult, QueryType, RetrievalStrategy
from rag_agent_platform.retrieval.mock import MockRetriever


@pytest.fixture
def agent() -> MockAgentService:
    return MockAgentService(
        naive_retriever=MockRetriever(retrieval_method="naive"),
        advanced_retriever=MockRetriever(retrieval_method="advanced"),
        graph_retriever=MockRetriever(retrieval_method="graph"),
    )


@pytest.mark.parametrize(
    ("query", "expected_type", "expected_strategy"),
    [
        ("公司的年假有几天？", QueryType.SIMPLE, RetrievalStrategy.NAIVE),
        ("请结合两份制度综合比较报销规则。", QueryType.COMPLEX, RetrievalStrategy.ADVANCED),
        ("采购部门和财务部门是什么关系？", QueryType.RELATION, RetrievalStrategy.GRAPH),
        ("你好", QueryType.CHAT, RetrievalStrategy.NONE),
    ],
)
def test_agent_mode_routes_by_rule(
    agent: MockAgentService,
    query: str,
    expected_type: QueryType,
    expected_strategy: RetrievalStrategy,
) -> None:
    result = agent.invoke(query, mode="agent")

    assert result.query_type is expected_type
    assert result.strategy is expected_strategy


def test_manual_mode_has_priority(agent: MockAgentService) -> None:
    result = agent.invoke("两个部门是什么关系？", mode="naive")

    assert result.strategy is RetrievalStrategy.NAIVE
    assert result.query_type is QueryType.SIMPLE


def test_returns_agent_result_with_required_trace(agent: MockAgentService) -> None:
    result = agent.invoke("年假有几天？")

    assert isinstance(result, AgentResult)
    assert {
        "analyze_query",
        "route_retriever",
        "retrieve",
        "generate",
        "evaluate",
        "finish",
    }.issubset(result.execution_trace)
