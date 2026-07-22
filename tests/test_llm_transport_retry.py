import json
from collections.abc import Callable
from urllib import error

import pytest

from rag_agent_platform.agent import LangGraphAgentService
from rag_agent_platform.agent.router import QueryAnalysis
from rag_agent_platform.evaluation import AnswerEvaluator, EvaluationDecision, EvaluationResult
from rag_agent_platform.generation import GroundedAnswerGenerator
from rag_agent_platform.llm import LLMRequestError, LLMTransportError, OpenAICompatibleChatModel
from rag_agent_platform.models import QueryType, RetrievedChunk
from rag_agent_platform.retrieval import BaseRetriever


class FakeResponse:
    def __init__(self, content: str) -> None:
        self._body = json.dumps(
            {"choices": [{"message": {"content": content}}]}, ensure_ascii=False
        ).encode()

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return self._body


class SequenceOpener:
    def __init__(self, outcomes: list[object]) -> None:
        self.outcomes = outcomes
        self.calls = 0

    def __call__(self, request: object, timeout: float) -> FakeResponse:
        del request, timeout
        outcome = self.outcomes[min(self.calls, len(self.outcomes) - 1)]
        self.calls += 1
        if isinstance(outcome, BaseException):
            raise outcome
        assert isinstance(outcome, FakeResponse)
        return outcome


def model(
    opener: Callable[..., FakeResponse],
    sleeps: list[float],
) -> OpenAICompatibleChatModel:
    return OpenAICompatibleChatModel(
        model="test-model",
        api_key="test-key",
        base_url="https://example.invalid/v1",
        max_transport_retries=2,
        urlopen_func=opener,
        sleep_func=sleeps.append,
    )


def test_connection_reset_is_retried_then_succeeds_without_real_sleep() -> None:
    opener = SequenceOpener(
        [ConnectionResetError(10054, "connection reset"), FakeResponse("重要采购 [1]")]
    )
    sleeps: list[float] = []

    answer = model(opener, sleeps).invoke("prompt")

    assert answer == "重要采购 [1]"
    assert opener.calls == 2
    assert sleeps == [0.5]


@pytest.mark.parametrize("status", [429, 500, 502, 503, 504])
def test_retryable_http_statuses_retry_to_the_limit(status: int) -> None:
    transient = error.HTTPError(
        "https://example.invalid/v1/chat/completions", status, "temporary", None, None
    )
    opener = SequenceOpener([transient])
    sleeps: list[float] = []

    with pytest.raises(LLMTransportError) as raised:
        model(opener, sleeps).invoke("prompt")

    assert raised.value.original_exception_type == "HTTPError"
    assert opener.calls == 3
    assert sleeps == [0.5, 1.0]


def test_unauthorized_is_not_retried() -> None:
    unauthorized = error.HTTPError(
        "https://example.invalid/v1/chat/completions", 401, "unauthorized", None, None
    )
    opener = SequenceOpener([unauthorized])
    sleeps: list[float] = []

    with pytest.raises(LLMRequestError) as raised:
        model(opener, sleeps).invoke("prompt")

    assert raised.value.status_code == 401
    assert opener.calls == 1
    assert sleeps == []


class ComplexAnalyzer:
    def analyze(self, query: str) -> QueryAnalysis:
        return QueryAnalysis(QueryType.COMPLEX, True, "fixture")


class OneChunkRetriever(BaseRetriever):
    def __init__(self, chunk: RetrievedChunk) -> None:
        self.chunk = chunk
        self.calls = 0

    def retrieve(
        self,
        query: str,
        document_ids: list[str] | None = None,
        top_k: int = 5,
    ) -> list[RetrievedChunk]:
        del query, document_ids, top_k
        self.calls += 1
        return [self.chunk]


class PassEvaluator(AnswerEvaluator):
    def evaluate(self, query: str, answer: str, chunks: list[RetrievedChunk]) -> EvaluationResult:
        del query, answer, chunks
        return EvaluationResult(
            True,
            "fixture pass",
            decision=EvaluationDecision.PASS,
            relevance_score=1.0,
            groundedness_score=1.0,
            completeness_score=1.0,
            citation_quality_score=1.0,
        )


def build_service(
    generator: GroundedAnswerGenerator,
    chunk: RetrievedChunk,
    evaluator: AnswerEvaluator,
) -> LangGraphAgentService:
    retriever = OneChunkRetriever(chunk)
    return LangGraphAgentService(
        naive_retriever=OneChunkRetriever(chunk),
        advanced_retriever=retriever,
        graph_retriever=OneChunkRetriever(chunk),
        generator=generator,
        evaluator=evaluator,
        analyzer=ComplexAnalyzer(),
    )


def test_transport_retry_does_not_increment_agent_retry_counters() -> None:
    opener = SequenceOpener(
        [ConnectionResetError(10054, "connection reset"), FakeResponse("重要采购 [1]")]
    )
    chat_model = model(opener, [])
    chunk = RetrievedChunk(
        "amount",
        "单笔采购金额超过五万元但不超过二十万元的，属于重要采购。",
        0.9,
        "policy.txt",
        retrieval_method="advanced",
    )

    result = build_service(GroundedAnswerGenerator(chat_model), chunk, PassEvaluator()).invoke(
        "八万块采购的流程怎么走？"
    )

    assert result.retry_count == 0
    assert result.regenerate_count == 0
    assert any("llm_transport_retry: retry=1/2" in step for step in result.execution_trace)
    assert any("llm_transport: success_after_retries=1" in step for step in result.execution_trace)


def test_exhausted_transport_uses_grounded_fallback_without_document_dump() -> None:
    policy = (
        "采购物品交付后，申请部门负责进行业务验收。信息技术设备还必须由信息安全部门参与安全验收。"
        "验收通过后，申请部门提交验收记录，采购部门提交合同和交付记录，供应商提交有效发票。"
        "财务部门收到完整材料后，应当在十个工作日内完成付款。"
        "如果验收不合格，在问题解决之前，财务部门不得付款。"
        "正式员工每年享有十天带薪年假。"
    )
    opener = SequenceOpener([ConnectionResetError(10054, "connection reset")])
    chat_model = model(opener, [])
    chunk = RetrievedChunk(
        "payment",
        policy,
        0.9,
        "policy.txt",
        retrieval_method="advanced",
    )

    result = build_service(
        GroundedAnswerGenerator(chat_model),
        chunk,
        PassEvaluator(),
    ).invoke("东西送来以后怎么才能打钱？")

    assert opener.calls == 3
    assert result.refused is False
    assert result.retry_count == 0
    assert result.regenerate_count == 0
    assert "业务验收" in result.answer
    assert "合同和交付记录" in result.answer
    assert "十个工作日" in result.answer
    assert "带薪年假" not in result.answer
    assert result.error is None
    assert any("llm_transport: failed_after_attempts=3" in step for step in result.execution_trace)
    assert any("deterministic_fallback" in step for step in result.execution_trace)
