from collections.abc import Iterable

import pytest

from rag_agent_platform.agent import LangGraphAgentService
from rag_agent_platform.agent.router import StructuredQueryAnalyzer
from rag_agent_platform.evaluation import AnswerEvaluator, EvaluationResult
from rag_agent_platform.generation import INSUFFICIENT_ANSWER, AnswerGenerator
from rag_agent_platform.models import (
    Citation,
    QueryType,
    RetrievalStrategy,
    RetrievedChunk,
)
from rag_agent_platform.retrieval import BaseRetriever


def evidence(method: str = "fake") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id="chunk-1",
        content="员工每年享有十天年假。",
        normalized_score=0.9,
        source="policy.txt",
        document_id="doc-1",
        parent_id="parent-1",
        page=2,
        retrieval_method=method,
    )


class RecordingRetriever(BaseRetriever):
    def __init__(self, chunks: list[RetrievedChunk] | None = None, *, fail: bool = False) -> None:
        self.chunks = chunks if chunks is not None else [evidence()]
        self.fail = fail
        self.calls: list[tuple[str, list[str] | None, int]] = []

    def retrieve(
        self,
        query: str,
        document_ids: list[str] | None = None,
        top_k: int = 5,
    ) -> list[RetrievedChunk]:
        self.calls.append((query, document_ids, top_k))
        if self.fail:
            raise ConnectionError("retriever unavailable")
        return list(self.chunks)


class FakeGenerator(AnswerGenerator):
    def generate(self, query: str, chunks: list[RetrievedChunk]) -> tuple[str, list[Citation]]:
        if not chunks:
            return INSUFFICIENT_ANSWER, []
        return (
            f"{query}：员工每年享有十天年假 [1]",
            [Citation(1, chunks[0].source, chunks[0].page, chunks[0].chunk_id)],
        )

    def generate_chat(self, query: str) -> str:
        return f"普通对话：{query}"


class SequenceEvaluator(AnswerEvaluator):
    def __init__(self, decisions: Iterable[EvaluationResult]) -> None:
        self.decisions = list(decisions)
        self.calls = 0

    def evaluate(self, query: str, answer: str, chunks: list[RetrievedChunk]) -> EvaluationResult:
        decision = self.decisions[min(self.calls, len(self.decisions) - 1)]
        self.calls += 1
        return decision


def build_agent(
    naive: RecordingRetriever | None = None,
    advanced: RecordingRetriever | None = None,
    graph: RecordingRetriever | None = None,
    evaluator: AnswerEvaluator | None = None,
    *,
    top_k: int = 3,
    max_retries: int = 2,
    analyzer: StructuredQueryAnalyzer | None = None,
) -> tuple[LangGraphAgentService, RecordingRetriever, RecordingRetriever, RecordingRetriever]:
    naive = naive or RecordingRetriever([evidence("naive")])
    advanced = advanced or RecordingRetriever([evidence("advanced")])
    graph = graph or RecordingRetriever([evidence("graph")])
    service = LangGraphAgentService(
        naive_retriever=naive,
        advanced_retriever=advanced,
        graph_retriever=graph,
        generator=FakeGenerator(),
        evaluator=evaluator or SequenceEvaluator([EvaluationResult(True, "passed")]),
        analyzer=analyzer,
        top_k=top_k,
        max_retries=max_retries,
    )
    return service, naive, advanced, graph


@pytest.mark.parametrize(
    ("mode", "expected_strategy", "selected_index"),
    [
        ("naive", RetrievalStrategy.NAIVE, 0),
        ("advanced", RetrievalStrategy.ADVANCED, 1),
        ("graph", RetrievalStrategy.GRAPH, 2),
    ],
)
def test_manual_modes_select_only_the_requested_retriever(
    mode: str, expected_strategy: RetrievalStrategy, selected_index: int
) -> None:
    service, naive, advanced, graph = build_agent()

    result = service.invoke("员工年假有几天？", ["doc-1"], mode)

    assert result.strategy is expected_strategy
    retrievers = [naive, advanced, graph]
    assert retrievers[selected_index].calls == [("员工年假有几天？", ["doc-1"], 3)]
    assert sum(len(retriever.calls) for retriever in retrievers) == 1
    assert isinstance(result.retrieved_chunks[0], RetrievedChunk)


@pytest.mark.parametrize(
    ("query", "query_type", "strategy", "selected_index"),
    [
        ("年假有几天？", QueryType.SIMPLE, RetrievalStrategy.NAIVE, 0),
        ("结合制度比较年假和病假的申请条件", QueryType.COMPLEX, RetrievalStrategy.ADVANCED, 1),
        ("采购部与供应商的上下游关系是什么？", QueryType.RELATION, RetrievalStrategy.GRAPH, 2),
    ],
)
def test_agent_mode_routes_knowledge_queries(
    query: str,
    query_type: QueryType,
    strategy: RetrievalStrategy,
    selected_index: int,
) -> None:
    service, naive, advanced, graph = build_agent()

    result = service.invoke(query, ["doc-1"], "agent")

    assert (result.query_type, result.strategy) == (query_type, strategy)
    assert len([naive, advanced, graph][selected_index].calls) == 1


def test_agent_mode_routes_chat_without_retrieval() -> None:
    service, naive, advanced, graph = build_agent()

    result = service.invoke("你好", mode="agent")

    assert result.query_type is QueryType.CHAT
    assert result.strategy is RetrievalStrategy.NONE
    assert result.citations == []
    assert not any(retriever.calls for retriever in (naive, advanced, graph))


def test_failed_evaluation_rewrites_then_second_attempt_passes() -> None:
    evaluator = SequenceEvaluator(
        [
            EvaluationResult(False, "缺少申请时限", "年假天数和申请时限"),
            EvaluationResult(True, "complete"),
        ]
    )
    service, naive, _, _ = build_agent(evaluator=evaluator)

    result = service.invoke("年假制度", ["doc-1"], "naive")

    assert result.retry_count == 1
    assert len(naive.calls) == 2
    assert naive.calls[1][0] == "年假天数和申请时限"
    assert any("rewrite_query" in step for step in result.execution_trace)


def test_workflow_stops_at_max_retries() -> None:
    evaluator = SequenceEvaluator([EvaluationResult(False, "证据不完整")])
    service, naive, _, _ = build_agent(evaluator=evaluator, max_retries=2)

    result = service.invoke("年假制度", ["doc-1"], "naive")

    assert result.answer == INSUFFICIENT_ANSWER
    assert result.retry_count == 2
    assert len(naive.calls) == 3
    assert any("达到重试上限" in step for step in result.execution_trace)


def test_citations_match_retrieved_chunks() -> None:
    service, _, _, _ = build_agent()

    result = service.invoke("年假有几天？", ["doc-1"], "naive")

    assert result.citations == [Citation(1, "policy.txt", 2, "chunk-1")]
    assert "[1]" in result.answer


def test_empty_context_never_generates_a_deterministic_answer() -> None:
    service, _, _, _ = build_agent(naive=RecordingRetriever([]), max_retries=0)

    result = service.invoke("未知政策", ["doc-1"], "naive")

    assert result.answer == INSUFFICIENT_ANSWER
    assert result.citations == []


def test_retriever_error_is_returned_clearly_without_retrying() -> None:
    failing = RecordingRetriever(fail=True)
    service, _, _, _ = build_agent(naive=failing)

    result = service.invoke("年假", ["doc-1"], "naive")

    assert result.error is not None
    assert "retriever unavailable" in result.error
    assert result.answer == INSUFFICIENT_ANSWER
    assert len(failing.calls) == 1


class FailingChatModel:
    def invoke(self, prompt: str) -> str:
        raise RuntimeError("classification unavailable")


def test_llm_classification_failure_uses_rule_fallback() -> None:
    analyzer = StructuredQueryAnalyzer(FailingChatModel())
    service, naive, _, _ = build_agent(analyzer=analyzer)

    result = service.invoke("年假有几天？", ["doc-1"], "agent")

    assert result.strategy is RetrievalStrategy.NAIVE
    assert naive.calls
    assert any("规则兜底" in step for step in result.execution_trace)


def test_invalid_mode_fails_fast() -> None:
    service, _, _, _ = build_agent()

    with pytest.raises(ValueError, match="unsupported mode"):
        service.invoke("年假", mode="unsupported")
