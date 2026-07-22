import pytest

from rag_agent_platform.agent.router import BoundedQueryRewriter, StructuredQueryAnalyzer
from rag_agent_platform.models import QueryType


class FixedChatModel:
    def __init__(self, response: str) -> None:
        self.response = response

    def invoke(self, prompt: str) -> str:
        return self.response


def rewrite(query: str, response: str | None = None) -> str:
    model = None if response is None else FixedChatModel(response)
    return BoundedQueryRewriter(model).rewrite(
        original_query=query,
        current_query=query,
        evaluation_reason="检索证据不足，需要保守规范化检索表达",
        chunks=[],
        suggested_query=None,
    )


def test_drifting_database_rewrite_is_rejected_and_falls_back_conservatively() -> None:
    result = rewrite(
        "八万块买数据库那个流程怎么走？",
        "采购Oracle或MySQL企业版许可证时，云帆信息技术有限公司是否具有授权代理资格？",
    )

    assert ("8万" in result or "80000" in result) and "数据库" in result
    assert "采购" in result and "流程" in result
    assert "Oracle" not in result
    assert "MySQL" not in result
    assert "许可证" not in result
    assert "授权代理" not in result
    assert "云帆" not in result
    assert "供应商资质" not in result
    assert StructuredQueryAnalyzer().analyze(result).query_type is not QueryType.RELATION


@pytest.mark.parametrize(
    ("query", "required", "forbidden"),
    [
        (
            "那种没预算又很急的情况要找谁？",
            ("未列入年度预算", "紧急采购", "谁", "批准"),
            ("金额", "供应商"),
        ),
        (
            "东西送来以后怎么才能打钱？",
            ("采购物品", "交付后", "验收", "材料", "付款"),
            ("总经理", "金额"),
        ),
    ],
)
def test_colloquial_queries_are_normalized_without_new_business_facts(
    query: str,
    required: tuple[str, ...],
    forbidden: tuple[str, ...],
) -> None:
    result = rewrite(query)

    assert all(term in result for term in required)
    assert all(term not in result for term in forbidden)


def test_unsafe_evaluator_suggestion_is_validated_too() -> None:
    rewriter = BoundedQueryRewriter()

    result = rewriter.rewrite(
        original_query="八万块买数据库那个流程怎么走？",
        current_query="八万块买数据库那个流程怎么走？",
        evaluation_reason="回答包含无依据推断",
        chunks=[],
        suggested_query="八万元购买MySQL许可证时云帆公司是否有代理资质？",
    )

    assert "MySQL" not in result
    assert "云帆" not in result
    assert "资质" not in result


def test_analyzer_rule_guard_blocks_relation_drift_for_amount_workflow() -> None:
    model = FixedChatModel(
        '{"query_type":"relation","needs_retrieval":true,"reason":"contains entities"}'
    )

    analysis = StructuredQueryAnalyzer(model).analyze("八万块买数据库那个流程怎么走？")

    assert analysis.query_type is QueryType.COMPLEX
    assert "策略漂移" in analysis.reason
