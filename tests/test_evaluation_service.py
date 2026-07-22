from pathlib import Path

from rag_agent_platform.evaluation import (
    EvaluationDecision,
    GroundedAnswerEvaluator,
)
from rag_agent_platform.generation import INSUFFICIENT_ANSWER
from rag_agent_platform.models import RetrievedChunk


def chunk(chunk_id: str, content: str, *, source: str = "采购制度.txt") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        content=content,
        normalized_score=0.9,
        source=source,
        document_id="policy",
        retrieval_method="fixture",
    )


AMOUNT = chunk(
    "amount",
    "单笔采购金额超过五万元但不超过二十万元的，属于重要采购。",
)
APPROVAL = chunk(
    "approval",
    "重要采购需要部门负责人、财务部门和分管副总经理依次审批，并至少比较三家供应商的报价。",
)
SECURITY = chunk(
    "security",
    "采购数据库软件时，申请必须先经过信息安全部门审核。",
)


class FixedEvaluationModel:
    def __init__(self, response: str) -> None:
        self.response = response

    def invoke(self, prompt: str) -> str:
        return self.response


def test_correct_grounded_answer_passes() -> None:
    result = GroundedAnswerEvaluator().evaluate(
        "八万块买数据库那个流程怎么走？",
        "8万元属于重要采购 [1]；数据库软件须先经信息安全部门审核 [3]；"
        "之后由部门负责人、财务部门和分管副总经理依次审批，并比较三家报价 [2]。",
        [AMOUNT, APPROVAL, SECURITY],
    )

    assert result.passed is True
    assert result.decision is EvaluationDecision.PASS


def test_supplier_hallucination_requires_regeneration_with_same_evidence() -> None:
    vendor = chunk(
        "vendor",
        "云帆信息技术有限公司主要提供服务器、网络设备和数据库维护服务。",
    )

    result = GroundedAnswerEvaluator().evaluate(
        "八万块买数据库那个流程怎么走？",
        "云帆信息技术有限公司可以销售数据库软件，并可作为本次供应商 [4]。",
        [AMOUNT, APPROVAL, SECURITY, vendor],
    )

    assert result.passed is False
    assert result.decision is EvaluationDecision.REGENERATE
    assert result.unsupported_claims


def test_missing_amount_rule_requires_rewrite_and_retrieval() -> None:
    result = GroundedAnswerEvaluator().evaluate(
        "八万块买数据库那个流程怎么走？",
        "数据库软件采购必须先经过信息安全部门审核 [1]。",
        [SECURITY],
    )

    assert result.decision is EvaluationDecision.REWRITE_RETRIEVE


def test_conflicting_procurement_level_requires_regeneration() -> None:
    result = GroundedAnswerEvaluator().evaluate(
        "八万块买数据库那个流程怎么走？",
        "8万元属于重大采购 [1]；数据库软件须先经过信息安全部门审核 [3]。",
        [AMOUNT, APPROVAL, SECURITY],
    )

    assert result.decision is EvaluationDecision.REGENERATE
    assert any("重大采购" in claim for claim in result.unsupported_claims)


def test_unrelated_evidence_refuses() -> None:
    result = GroundedAnswerEvaluator().evaluate(
        "公司是否报销员工的火星旅行费用？",
        INSUFFICIENT_ANSWER,
        [chunk("leave", "正式员工每年享有十天带薪年假。", source="休假制度.txt")],
    )

    assert result.decision is EvaluationDecision.REFUSE


def test_non_critical_paraphrase_and_same_document_citations_pass() -> None:
    controls = chunk(
        "controls",
        "信息安全部门主要检查访问控制、数据加密、日志留存和漏洞修复。",
    )
    qualification = chunk(
        "qualification",
        "信息安全部门还检查供应商安全资质。",
    )

    result = GroundedAnswerEvaluator().evaluate(
        "信息安全审核检查什么？",
        "审核检查访问控制、数据加密、日志留存和漏洞修复能力 [1]，也检查供应商安全资质 [2]。",
        [controls, qualification],
    )

    assert result.decision is EvaluationDecision.PASS
    assert result.citation_quality_score > 0


def test_overstrict_model_cannot_turn_sufficient_evidence_into_retrieval_rewrite() -> None:
    model = FixedEvaluationModel(
        '{"decision":"rewrite_retrieve","reason":"未区分商业许可证类型",'
        '"relevance_score":0.9,"groundedness_score":0.9,"completeness_score":0.4,'
        '"citation_quality_score":0.9,"suggested_query":"Oracle许可证采购流程",'
        '"unsupported_claims":[]}'
    )

    result = GroundedAnswerEvaluator(model).evaluate(
        "八万块买数据库那个流程怎么走？",
        "8万元属于重要采购 [1]；数据库软件先经信息安全部门审核 [3]；"
        "然后按重要采购层级依次审批并比较三家报价 [2]。",
        [AMOUNT, APPROVAL, SECURITY],
    )

    assert result.decision is EvaluationDecision.REGENERATE
    assert result.suggested_query is None


def test_long_semantically_corrupted_rewrite_requires_regeneration() -> None:
    policy = (Path(__file__).parent / "fixtures" / "enterprise_procurement_policy.txt").read_text(
        encoding="utf-8"
    )
    evidence = chunk("parent", policy)
    corrupted = (
        "8万元属于重要采购 [1]。数据库软件先由信息安全部门审核 [1]。"
        "随后由行政部门负责人赫尔、财务屋顶和仓储马尼拉、分管副总经理依次递推 [1]。"
        "紧急采购完成后三个工作日内补齐天线，由直属人猿负责 [1]。"
        "正式员工的年假普遍采用十天，并补充病假和采购合同的全部章节 [1]。"
    )

    result = GroundedAnswerEvaluator().evaluate(
        "八万块买数据库那个流程怎么走？",
        corrupted,
        [evidence],
    )

    assert result.passed is False
    assert result.decision is EvaluationDecision.REGENERATE
    assert result.unsupported_claims
