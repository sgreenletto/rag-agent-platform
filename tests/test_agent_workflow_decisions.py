from collections.abc import Iterable

from rag_agent_platform.agent import LangGraphAgentService
from rag_agent_platform.agent.router import QueryAnalysis
from rag_agent_platform.evaluation import AnswerEvaluator, EvaluationDecision, EvaluationResult
from rag_agent_platform.generation import INSUFFICIENT_ANSWER, AnswerGenerator
from rag_agent_platform.models import Citation, QueryType, RetrievalStrategy, RetrievedChunk
from rag_agent_platform.retrieval import BaseRetriever


def evidence(chunk_id: str = "one", content: str = "重要采购由部门负责人审批。") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        content=content,
        normalized_score=0.9,
        source="policy.txt",
        document_id="policy",
        retrieval_method="advanced",
    )


class ComplexAnalyzer:
    def analyze(self, query: str) -> QueryAnalysis:
        return QueryAnalysis(QueryType.COMPLEX, True, "fixture complex query")


class RecordingRetriever(BaseRetriever):
    def __init__(self, responses: list[list[RetrievedChunk]] | None = None) -> None:
        self.responses = responses or [[evidence()]]
        self.calls: list[str] = []

    def retrieve(
        self,
        query: str,
        document_ids: list[str] | None = None,
        top_k: int = 5,
    ) -> list[RetrievedChunk]:
        self.calls.append(query)
        return list(self.responses[min(len(self.calls) - 1, len(self.responses) - 1)])


class RecordingGenerator(AnswerGenerator):
    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str]]] = []

    def generate(self, query: str, chunks: list[RetrievedChunk]) -> tuple[str, list[Citation]]:
        self.calls.append((query, [item.chunk_id for item in chunks]))
        answer = "；".join(f"{item.content} [{index}]" for index, item in enumerate(chunks, 1))
        citations = [
            Citation(index, item.source, item.page, item.chunk_id)
            for index, item in enumerate(chunks, 1)
        ]
        return answer, citations


class SequenceEvaluator(AnswerEvaluator):
    def __init__(self, decisions: Iterable[EvaluationResult]) -> None:
        self.decisions = list(decisions)
        self.calls = 0

    def evaluate(self, query: str, answer: str, chunks: list[RetrievedChunk]) -> EvaluationResult:
        result = self.decisions[min(self.calls, len(self.decisions) - 1)]
        self.calls += 1
        return result


def decision(kind: EvaluationDecision, *, suggested_query: str | None = None) -> EvaluationResult:
    return EvaluationResult(
        passed=kind is EvaluationDecision.PASS,
        reason=f"fixture {kind.value}",
        suggested_query=suggested_query,
        decision=kind,
        relevance_score=0.9,
        groundedness_score=0.8,
        completeness_score=0.8,
        citation_quality_score=0.8,
    )


def build_service(
    retriever: RecordingRetriever,
    generator: RecordingGenerator,
    evaluator: AnswerEvaluator,
    *,
    max_retries: int = 2,
    max_regenerations: int = 1,
) -> LangGraphAgentService:
    return LangGraphAgentService(
        naive_retriever=RecordingRetriever(),
        advanced_retriever=retriever,
        graph_retriever=RecordingRetriever(),
        generator=generator,
        evaluator=evaluator,
        analyzer=ComplexAnalyzer(),
        max_retries=max_retries,
        max_regenerations=max_regenerations,
    )


def test_regenerate_reuses_evidence_query_and_strategy() -> None:
    retriever = RecordingRetriever()
    generator = RecordingGenerator()
    service = build_service(
        retriever,
        generator,
        SequenceEvaluator(
            [decision(EvaluationDecision.REGENERATE), decision(EvaluationDecision.PASS)]
        ),
    )

    result = service.invoke("八万块买数据库那个流程怎么走？")

    assert len(generator.calls) == 2
    assert len(retriever.calls) == 1
    assert result.retry_count == 0
    assert result.regenerate_count == 1
    assert result.query_history == ["八万块买数据库那个流程怎么走？"]
    assert result.strategy_history == [RetrievalStrategy.ADVANCED]
    assert any("decision=regenerate" in step for step in result.execution_trace)


def test_rewrite_retrieve_replaces_evidence_and_counts_only_retrieval_retry() -> None:
    first = evidence("first", "只有数据库软件需要信息安全审核。")
    second = evidence("second", "八万元属于重要采购。")
    retriever = RecordingRetriever([[first], [second]])
    generator = RecordingGenerator()
    service = build_service(
        retriever,
        generator,
        SequenceEvaluator(
            [
                decision(
                    EvaluationDecision.REWRITE_RETRIEVE,
                    suggested_query="金额八万元的数据库软件采购审批流程",
                ),
                decision(EvaluationDecision.PASS),
            ]
        ),
    )

    result = service.invoke("八万块买数据库那个流程怎么走？")

    assert len(retriever.calls) == 2
    assert retriever.calls[1] != retriever.calls[0]
    assert result.retry_count == 1
    assert result.regenerate_count == 0
    assert [item.chunk_id for item in result.retrieved_chunks] == ["second"]
    assert len(result.query_history) == 2
    assert result.strategy_history == [RetrievalStrategy.ADVANCED, RetrievalStrategy.ADVANCED]


def test_retrieval_retry_limit_refuses_without_infinite_loop() -> None:
    retriever = RecordingRetriever()
    generator = RecordingGenerator()
    service = build_service(
        retriever,
        generator,
        SequenceEvaluator([decision(EvaluationDecision.REWRITE_RETRIEVE)]),
        max_retries=2,
    )

    result = service.invoke("八万块买数据库那个流程怎么走？")

    assert len(retriever.calls) == 3
    assert result.retry_count == 2
    assert result.refused is True
    assert result.answer == INSUFFICIENT_ANSWER


def test_regeneration_limit_uses_bounded_conservative_answer() -> None:
    retriever = RecordingRetriever()
    generator = RecordingGenerator()
    service = build_service(
        retriever,
        generator,
        SequenceEvaluator([decision(EvaluationDecision.REGENERATE)]),
        max_regenerations=1,
    )

    result = service.invoke("八万块买数据库那个流程怎么走？")

    assert len(generator.calls) == 2
    assert len(retriever.calls) == 1
    assert result.regenerate_count == 1
    assert result.retry_count == 0
    assert result.refused is False
    assert "重要采购" in result.answer
    assert any("重新生成上限" in step for step in result.execution_trace)


def test_clarify_and_refuse_are_distinct_terminal_paths() -> None:
    clarify_service = build_service(
        RecordingRetriever(),
        RecordingGenerator(),
        SequenceEvaluator(
            [
                EvaluationResult(
                    False,
                    "缺少会改变制度结论的采购类型",
                    "请确认采购对象属于设备还是服务？",
                    decision=EvaluationDecision.CLARIFY,
                )
            ]
        ),
    )
    refuse_service = build_service(
        RecordingRetriever(),
        RecordingGenerator(),
        SequenceEvaluator([decision(EvaluationDecision.REFUSE)]),
    )

    clarified = clarify_service.invoke("怎么买？")
    refused = refuse_service.invoke("火星旅行费用报销吗？")

    assert "请确认" in clarified.answer
    assert clarified.refused is False
    assert refused.answer == INSUFFICIENT_ANSWER
    assert refused.refused is True


def test_generator_and_citations_use_final_deduplicated_chunk_order() -> None:
    duplicate_a = evidence("child-a", "重要采购需要部门负责人依次审批。")
    duplicate_a.parent_id = "parent"
    duplicate_a.metadata["parent_context_chunk_id"] = "parent"
    duplicate_b = evidence("child-b", "重要采购需要部门负责人依次审批。")
    duplicate_b.parent_id = "parent"
    duplicate_b.metadata["parent_context_chunk_id"] = "parent"
    distinct = evidence("security", "数据库软件采购必须先经过信息安全部门审核。")
    retriever = RecordingRetriever([[duplicate_a, duplicate_b, distinct]])
    generator = RecordingGenerator()
    service = build_service(
        retriever,
        generator,
        SequenceEvaluator([decision(EvaluationDecision.PASS)]),
    )

    result = service.invoke("八万块买数据库那个流程怎么走？")

    assert generator.calls == [("八万块买数据库那个流程怎么走？", ["child-a", "security"])]
    assert [citation.index for citation in result.citations] == [1, 2]
    assert [citation.chunk_id for citation in result.citations] == ["child-a", "security"]
