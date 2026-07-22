from pathlib import Path

from rag_agent_platform.generation.fallback import GroundedFallbackSynthesizer
from rag_agent_platform.models import RetrievalStrategy, RetrievedChunk

FIXTURE = Path(__file__).parent / "fixtures" / "enterprise_procurement_policy.txt"


def parent_chunk(content: str, *, method: str = "advanced") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id="policy:parent:1",
        content=content,
        normalized_score=0.95,
        source="enterprise_procurement_policy.txt",
        document_id="policy",
        retrieval_method=method,
    )


def test_fallback_selects_relevant_sentences_instead_of_dumping_long_parent() -> None:
    full_policy = FIXTURE.read_text(encoding="utf-8")

    result = GroundedFallbackSynthesizer(max_answer_chars=700).synthesize(
        "八万块买数据库那个流程怎么走？",
        [parent_chunk(full_policy)],
        retrieval_strategy=RetrievalStrategy.ADVANCED,
    )

    assert result.sufficient is True
    assert len(result.answer) <= 700
    assert len(result.answer) < len(full_policy) // 2
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
    assert "员工休假" not in result.answer
    assert "带薪年假" not in result.answer
    assert [item.index for item in result.citations] == [1]
    assert result.citations[0].chunk_id == "policy:parent:1"


def test_payment_fallback_covers_required_records_and_no_unrelated_chapters() -> None:
    full_policy = FIXTURE.read_text(encoding="utf-8")

    result = GroundedFallbackSynthesizer().synthesize(
        "东西送来以后怎么才能打钱？",
        [parent_chunk(full_policy)],
        retrieval_strategy=RetrievalStrategy.ADVANCED,
    )

    for fact in (
        "业务验收",
        "安全验收",
        "验收记录",
        "合同和交付记录",
        "有效发票",
        "十个工作日",
        "不得付款",
    ):
        assert fact in result.answer
    assert "采购金额分级" not in result.answer
    assert "员工休假" not in result.answer
    assert len(result.answer) <= 700


def test_graph_fallback_prefers_relation_evidence_and_capability_boundary() -> None:
    chunk = parent_chunk(
        "云帆信息技术有限公司是星海科技有限公司的信息技术供应商，主要提供服务器、"
        "网络设备和数据库维护服务。采购部门负责联系华星办公用品有限公司和云帆信息技术"
        "有限公司，并负责与供应商签订采购合同。正式员工每年享有十天带薪年假。",
        method="graph",
    )
    chunk.metadata["graph_relations"] = [
        {
            "subject": "采购部门",
            "predicate": "负责联系",
            "object": "云帆信息技术有限公司",
        },
        {
            "subject": "采购部门",
            "predicate": "负责签订",
            "object": "采购合同",
        },
    ]

    result = GroundedFallbackSynthesizer().synthesize(
        "采购部门与云帆信息技术有限公司存在什么关联关系？",
        [chunk],
        retrieval_strategy=RetrievalStrategy.GRAPH,
    )

    assert result.sufficient is True
    assert "采购部门" in result.answer
    assert "联系" in result.answer
    assert "签订采购合同" in result.answer
    assert "华星办公用品有限公司" not in result.answer
    assert "数据库软件销售" not in result.answer
    assert "代理资质" not in result.answer
    assert "带薪年假" not in result.answer
    assert len(result.answer) <= 700


def test_fallback_refuses_when_evidence_is_unrelated() -> None:
    result = GroundedFallbackSynthesizer().synthesize(
        "公司是否报销员工的火星旅行费用？",
        [parent_chunk("正式员工每年享有十天带薪年假。")],
        retrieval_strategy=RetrievalStrategy.NAIVE,
    )

    assert result.sufficient is False
    assert result.citations == []


def test_fallback_citations_keep_original_chunk_indexes_after_sentence_selection() -> None:
    irrelevant = parent_chunk("正式员工每年享有十天带薪年假。")
    relevant = RetrievedChunk(
        chunk_id="policy:parent:2",
        content="财务部门收到完整材料后，应当在十个工作日内完成付款。",
        normalized_score=0.9,
        source="enterprise_procurement_policy.txt",
        document_id="policy",
        retrieval_method="advanced",
    )

    result = GroundedFallbackSynthesizer().synthesize(
        "材料齐了以后多久付款？",
        [irrelevant, relevant],
        retrieval_strategy=RetrievalStrategy.ADVANCED,
    )

    assert "十个工作日" in result.answer
    assert "[2]" in result.answer
    assert [citation.index for citation in result.citations] == [2]
    assert result.citations[0].chunk_id == "policy:parent:2"
