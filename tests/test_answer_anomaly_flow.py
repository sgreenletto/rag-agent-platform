from pathlib import Path

from rag_agent_platform.agent import LangGraphAgentService
from rag_agent_platform.agent.router import QueryAnalysis
from rag_agent_platform.evaluation import GroundedAnswerEvaluator
from rag_agent_platform.generation import AnswerGenerator
from rag_agent_platform.models import Citation, QueryType, RetrievedChunk
from rag_agent_platform.retrieval import BaseRetriever


class ComplexAnalyzer:
    def analyze(self, query: str) -> QueryAnalysis:
        return QueryAnalysis(QueryType.COMPLEX, True, "fixture")


class PolicyRetriever(BaseRetriever):
    def __init__(self, content: str) -> None:
        self.chunk = RetrievedChunk(
            "parent",
            content,
            0.95,
            "enterprise_procurement_policy.txt",
            retrieval_method="advanced",
        )
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


class CorruptingGenerator(AnswerGenerator):
    def __init__(self) -> None:
        self.calls = 0

    def generate(self, query: str, chunks: list[RetrievedChunk]) -> tuple[str, list[Citation]]:
        del query
        self.calls += 1
        answer = (
            "8万元属于重要采购 [1]。数据库软件先由信息安全部门审核 [1]。"
            "随后由行政部门负责人赫尔、财务屋顶和仓储马尼拉、分管副总经理依次递推 [1]。"
            "紧急采购完成后三个工作日内补齐天线，由直属人猿负责 [1]。"
            "正式员工的年假普遍采用十天，并复述所有无关章节 [1]。"
        )
        return answer, [Citation(1, chunks[0].source, chunks[0].page, chunks[0].chunk_id)]


def test_corrupted_generation_regenerates_once_then_uses_grounded_fallback() -> None:
    policy = (Path(__file__).parent / "fixtures" / "enterprise_procurement_policy.txt").read_text(
        encoding="utf-8"
    )
    retriever = PolicyRetriever(policy)
    generator = CorruptingGenerator()
    service = LangGraphAgentService(
        naive_retriever=PolicyRetriever(policy),
        advanced_retriever=retriever,
        graph_retriever=PolicyRetriever(policy),
        generator=generator,
        evaluator=GroundedAnswerEvaluator(),
        analyzer=ComplexAnalyzer(),
        max_regenerations=1,
    )

    result = service.invoke("八万块买数据库那个流程怎么走？")

    assert generator.calls == 2
    assert retriever.calls == 1
    assert result.regenerate_count == 1
    assert result.retry_count == 0
    assert result.refused is False
    for fact in (
        "8万元",
        "重要采购",
        "信息安全部门",
        "部门负责人",
        "财务部门",
        "分管副总经理",
        "三家供应商",
    ):
        assert fact in result.answer
    for unsupported in ("负责人赫尔", "仓储马尼拉", "依次递推", "补齐天线", "直属人猿"):
        assert unsupported not in result.answer
    assert "带薪年假" not in result.answer
    assert len(result.answer) <= 700
    assert any("deterministic_fallback" in step for step in result.execution_trace)
