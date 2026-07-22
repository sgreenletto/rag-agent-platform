from pathlib import Path

from rag_agent_platform.bootstrap import build_application_services
from rag_agent_platform.config import Settings
from rag_agent_platform.generation import INSUFFICIENT_ANSWER
from rag_agent_platform.models import RetrievalStrategy


def test_real_local_container_answers_policy_workflows_and_refuses_unknown(
    tmp_path: Path,
) -> None:
    config = Settings(
        _env_file=None,
        app_mode="real",
        document_repository_provider="file",
        embedding_provider="hash",
        llm_provider="",
        metadata_path=str(tmp_path / "metadata" / "documents.json"),
        chroma_persist_directory=str(tmp_path / "chroma"),
        chroma_collection_name="policy_e2e",
        graph_persist_directory=str(tmp_path / "graph"),
        upload_directory=str(tmp_path / "uploads"),
        retrieval_top_k=8,
        agent_max_retries=2,
        agent_max_regenerations=1,
    )
    services = build_application_services(config)
    fixture = Path(__file__).parent / "fixtures" / "enterprise_procurement_policy.txt"
    ingested = services.ingestion.ingest(fixture)
    document_ids = [ingested.document.document_id]

    database = services.agent.invoke("八万块买数据库那个流程怎么走？", document_ids, "agent")
    urgent = services.agent.invoke("那种没预算又很急的情况要找谁？", document_ids, "agent")
    payment = services.agent.invoke("东西送来以后怎么才能打钱？", document_ids, "agent")
    unknown = services.agent.invoke("公司是否报销员工的火星旅行费用？", document_ids, "agent")
    relation = services.agent.invoke(
        "采购部门与云帆信息技术有限公司存在什么关联？", document_ids, "agent"
    )

    assert database.strategy is RetrievalStrategy.ADVANCED
    assert RetrievalStrategy.GRAPH not in database.strategy_history
    assert database.refused is False
    assert database.retry_count == 0 and database.regenerate_count == 0
    for fact in (
        "重要采购",
        "信息安全部门",
        "部门负责人",
        "财务部门",
        "分管副总经理",
        "三家供应商",
    ):
        assert fact in database.answer
    assert all(
        term not in database.answer
        for term in ("Oracle", "MySQL 企业版", "授权代理资格", "数据库软件销售商")
    )
    assert urgent.refused is False
    assert urgent.retry_count == 0 and urgent.regenerate_count == 0
    assert "总经理批准" in urgent.answer and "三个工作日内补齐" in urgent.answer
    assert payment.refused is False
    assert payment.retry_count == 0 and payment.regenerate_count == 0
    for fact in (
        "业务验收",
        "安全验收",
        "验收记录",
        "合同和交付记录",
        "有效发票",
        "十个工作日",
    ):
        assert fact in payment.answer
    assert unknown.refused is True and unknown.answer == INSUFFICIENT_ANSWER
    assert unknown.retry_count == 0 and unknown.regenerate_count == 0
    assert relation.strategy is RetrievalStrategy.GRAPH
    assert relation.retry_count == 0 and relation.regenerate_count == 0
    assert "采购部门负责联系" in relation.answer
    assert "数据库软件销售" not in relation.answer
