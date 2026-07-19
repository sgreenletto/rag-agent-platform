"""Streamlit application assembly for the Mock project skeleton."""

import streamlit as st

from rag_agent_platform.agent import MockAgentService
from rag_agent_platform.config import settings
from rag_agent_platform.ingestion import MockIngestionPipeline
from rag_agent_platform.models import DocumentRecord
from rag_agent_platform.retrieval import MockRetriever
from rag_agent_platform.storage import MockDocumentRepository
from rag_agent_platform.ui.components import (
    render_agent_result,
    render_chat_history,
    render_sidebar,
)
from rag_agent_platform.ui.session import initialize_session_state


@st.cache_resource
def build_mock_services() -> tuple[MockIngestionPipeline, MockAgentService]:
    """Create process-local Mock services shared across Streamlit reruns."""

    repository = MockDocumentRepository()
    repository.save_document(
        DocumentRecord(
            document_id="mock-document-1",
            filename="mock-policy.txt",
            file_type="txt",
            source_path="mock://mock-policy.txt",
            status="mock-ready",
            metadata={"mock": True},
        )
    )
    repository.save_document(
        DocumentRecord(
            document_id="mock-document-2",
            filename="mock-handbook.md",
            file_type="md",
            source_path="mock://mock-handbook.md",
            status="mock-ready",
            metadata={"mock": True},
        )
    )
    ingestion = MockIngestionPipeline(repository)
    agent = MockAgentService(
        naive_retriever=MockRetriever(retrieval_method="naive"),
        advanced_retriever=MockRetriever(retrieval_method="advanced"),
        graph_retriever=MockRetriever(retrieval_method="graph"),
    )
    return ingestion, agent


def run_app() -> None:
    """Configure and render the complete Streamlit application."""

    st.set_page_config(
        page_title="模块化智能文档问答系统",
        page_icon="📚",
        layout="wide",
    )
    initialize_session_state()
    ingestion, agent = build_mock_services()
    selected_document_ids, current_mode = render_sidebar(ingestion)

    st.title(settings.app_name)
    st.subheader("模块化智能文档问答系统")
    st.warning(
        "当前为工程骨架和 Mock 阶段：仅用于验证接口、路由与 UI 闭环，"
        "尚未接入真实 LLM、Embedding、数据库或完整 LangGraph Agent。"
    )
    render_chat_history()

    query = st.chat_input("请输入你的问题")
    if not query:
        return

    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    try:
        result = agent.invoke(
            query=query,
            document_ids=selected_document_ids,
            mode=current_mode,
        )
    except (KeyError, TypeError, ValueError) as exc:
        st.error(f"Mock 问答执行失败：{exc}")
        return

    st.session_state.messages.append(
        {"role": "assistant", "content": result.answer, "result": result}
    )
    with st.chat_message("assistant"):
        render_agent_result(result)
