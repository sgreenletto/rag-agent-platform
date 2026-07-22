from pathlib import Path

import pytest

from rag_agent_platform.agent import LangGraphAgentService
from rag_agent_platform.agent.router import QueryAnalysis
from rag_agent_platform.evaluation import GroundedAnswerEvaluator
from rag_agent_platform.generation import INSUFFICIENT_ANSWER, GroundedAnswerGenerator
from rag_agent_platform.models import QueryType, RetrievalStrategy, RetrievedChunk
from rag_agent_platform.retrieval import BaseRetriever


@pytest.fixture(scope="module")
def policy_paragraphs() -> list[str]:
    path = Path(__file__).parent / "fixtures" / "enterprise_procurement_policy.txt"
    return [part.strip() for part in path.read_text(encoding="utf-8").split("\n\n") if part.strip()]


class PolicyAnalyzer:
    def analyze(self, query: str) -> QueryAnalysis:
        if "关联" in query or "关系" in query:
            return QueryAnalysis(QueryType.RELATION, True, "explicit relationship")
        if "数据库" in query or "没预算" in query or "打钱" in query:
            return QueryAnalysis(QueryType.COMPLEX, True, "multi-step policy workflow")
        return QueryAnalysis(QueryType.SIMPLE, True, "direct fact")


class FixturePolicyRetriever(BaseRetriever):
    def __init__(self, paragraphs: list[str], method: str) -> None:
        self.paragraphs = paragraphs
        self.method = method
        self.calls: list[str] = []

    def retrieve(
        self,
        query: str,
        document_ids: list[str] | None = None,
        top_k: int = 5,
    ) -> list[RetrievedChunk]:
        self.calls.append(query)
        groups: list[tuple[tuple[str, ...], tuple[str, ...]]] = [
            (
                ("数据库",),
                (
                    "超过五万元但不超过二十万元",
                    "重要采购需要部门负责人",
                    "数据库软件、云计算服务",
                    "信息安全部门审核通过后",
                ),
            ),
            (("没预算", "未列入年度预算"), ("未列入年度预算的紧急采购", "三个工作日")),
            (("打钱", "付款"), ("采购物品交付后", "验收通过后", "十个工作日")),
            (("关联", "关系"), ("采购部门负责联系华星",)),
            (("年假",), ("正式员工每年享有十天带薪年假",)),
        ]
        needles = next(
            (selected for triggers, selected in groups if any(item in query for item in triggers)),
            (),
        )
        selected = [
            paragraph
            for paragraph in self.paragraphs
            if any(needle in paragraph for needle in needles)
        ]
        return [
            RetrievedChunk(
                chunk_id=f"{self.method}-{index}",
                content=content,
                normalized_score=max(0.5, 1.0 - index * 0.05),
                source="enterprise_procurement_policy.txt",
                document_id="policy",
                retrieval_method=self.method,
            )
            for index, content in enumerate(selected[:top_k], 1)
        ]


@pytest.fixture
def policy_agent(
    policy_paragraphs: list[str],
) -> tuple[LangGraphAgentService, dict[str, FixturePolicyRetriever]]:
    retrievers = {
        "naive": FixturePolicyRetriever(policy_paragraphs, "naive"),
        "advanced": FixturePolicyRetriever(policy_paragraphs, "advanced"),
        "graph": FixturePolicyRetriever(policy_paragraphs, "graph"),
    }
    service = LangGraphAgentService(
        naive_retriever=retrievers["naive"],
        advanced_retriever=retrievers["advanced"],
        graph_retriever=retrievers["graph"],
        generator=GroundedAnswerGenerator(),
        evaluator=GroundedAnswerEvaluator(),
        analyzer=PolicyAnalyzer(),
    )
    return service, retrievers


def test_fuzzy_database_purchase_is_answered_without_strategy_drift(
    policy_agent: tuple[LangGraphAgentService, dict[str, FixturePolicyRetriever]],
) -> None:
    service, _ = policy_agent

    result = service.invoke("八万块买数据库那个流程怎么走？")

    assert result.refused is False
    assert result.strategy is RetrievalStrategy.ADVANCED
    assert RetrievalStrategy.GRAPH not in result.strategy_history
    expected_facts = (
        "超过五万元",
        "重要采购",
        "信息安全部门",
        "部门负责人",
        "财务部门",
        "分管副总经理",
        "三家供应商",
    )
    for fact in expected_facts:
        assert fact in result.answer
    for unsupported in ("Oracle", "MySQL 企业版", "云帆是数据库软件销售商", "授权代理资格"):
        assert unsupported not in result.answer
    assert result.regenerate_count <= 1


@pytest.mark.parametrize(
    ("query", "required"),
    [
        ("那种没预算又很急的情况要找谁？", ("总经理批准", "三个工作日内补齐")),
        (
            "东西送来以后怎么才能打钱？",
            ("业务验收", "信息安全部门", "验收记录", "合同和交付记录", "有效发票", "十个工作日"),
        ),
        ("正式员工每年有多少天带薪年假？", ("十天带薪年假",)),
    ],
)
def test_other_policy_questions_remain_answerable(
    policy_agent: tuple[LangGraphAgentService, dict[str, FixturePolicyRetriever]],
    query: str,
    required: tuple[str, ...],
) -> None:
    service, _ = policy_agent

    result = service.invoke(query)

    assert result.refused is False
    assert all(item in result.answer for item in required)


def test_no_answer_and_explicit_graph_relation_keep_their_terminal_behavior(
    policy_agent: tuple[LangGraphAgentService, dict[str, FixturePolicyRetriever]],
) -> None:
    service, _ = policy_agent

    no_answer = service.invoke("公司是否报销员工的火星旅行费用？")
    relation = service.invoke("采购部门与云帆信息技术有限公司存在什么关联？")

    assert no_answer.refused is True
    assert no_answer.answer == INSUFFICIENT_ANSWER
    assert relation.strategy is RetrievalStrategy.GRAPH
    assert "采购部门负责联系" in relation.answer
    assert "数据库软件销售" not in relation.answer
    assert "授权代理" not in relation.answer
