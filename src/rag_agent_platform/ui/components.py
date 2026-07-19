"""Reusable Streamlit presentation components."""

import hashlib
from pathlib import Path
from typing import Any

import streamlit as st

from rag_agent_platform.ingestion.base import IngestionPipeline
from rag_agent_platform.models import AgentResult


def render_sidebar(ingestion: IngestionPipeline) -> tuple[list[str], str]:
    """Render upload, document selection and mode controls."""

    with st.sidebar:
        st.title("RAG Agent Platform")
        st.caption("四人小组课程项目 · Mock 工程骨架")

        uploads = st.file_uploader(
            "上传文档",
            type=["txt", "md", "pdf", "docx"],
            accept_multiple_files=True,
            help="当前仅真实读取 TXT 和 Markdown；PDF、DOCX 只预留接口。",
        )
        _process_uploads(uploads, ingestion)

        documents = ingestion.list_documents()
        st.subheader("已有文档")
        if documents:
            for document in documents:
                st.caption(f"• {document.filename} · {document.status}")
        else:
            st.caption("尚无文档")

        document_names = {document.document_id: document.filename for document in documents}
        available_ids = list(document_names)
        st.session_state.selected_document_ids = [
            document_id
            for document_id in st.session_state.selected_document_ids
            if document_id in document_names
        ]
        selected_document_ids = st.multiselect(
            "选择知识源",
            options=available_ids,
            format_func=lambda document_id: document_names[document_id],
            key="selected_document_ids",
        )

        mode_labels = {
            "agent": "Agent 自动路由（Mock）",
            "naive": "Naive RAG（Mock）",
            "advanced": "Advanced RAG（Mock）",
            "graph": "GraphRAG（Mock）",
        }
        current_mode = st.radio(
            "问答模式",
            options=list(mode_labels),
            format_func=lambda value: mode_labels[value],
            key="current_mode",
        )

        if st.button("清空聊天", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    return selected_document_ids, current_mode


def render_chat_history() -> None:
    """Render all user and assistant messages saved in this session."""

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            result = message.get("result")
            if isinstance(result, AgentResult):
                render_agent_result(result)
            else:
                st.markdown(message["content"])


def render_agent_result(result: AgentResult) -> None:
    """Render a Mock answer and its structured execution details."""

    st.markdown(result.answer)
    with st.expander("执行信息与轨迹"):
        st.write(f"query_type：`{result.query_type.value}`")
        st.write(f"strategy：`{result.strategy.value}`")
        st.write(f"retry_count：`{result.retry_count}`")
        st.write("execution_trace：")
        for step in result.execution_trace:
            st.write(f"- {step}")

    with st.expander("来源与引用"):
        if not result.citations:
            st.caption("本次回答没有检索来源。")
        for citation in result.citations:
            page = f"，第 {citation.page} 页" if citation.page is not None else ""
            chunk = f"，chunk={citation.chunk_id}" if citation.chunk_id else ""
            st.write(f"[{citation.index}] {citation.source}{page}{chunk}")


def _process_uploads(uploads: list[Any], ingestion: IngestionPipeline) -> None:
    upload_dir = Path("data/uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)
    for upload in uploads:
        content = upload.getvalue()
        fingerprint = hashlib.sha256(upload.name.encode("utf-8") + content).hexdigest()
        if fingerprint in st.session_state.processed_uploads:
            continue

        suffix = Path(upload.name).suffix.lower()
        if suffix in {".pdf", ".docx"}:
            st.info(f"{upload.name}：接口已预留，解析器尚未实现。")
            st.session_state.processed_uploads.add(fingerprint)
            continue

        safe_name = Path(upload.name).name
        target = upload_dir / f"{fingerprint[:8]}-{safe_name}"
        try:
            target.write_bytes(content)
            result = ingestion.ingest(target)
        except (OSError, UnicodeDecodeError, ValueError) as exc:
            st.error(f"{upload.name} 入库失败：{exc}")
        else:
            st.success(
                f"{upload.name} 已完成 Mock 入库："
                f"{result.parent_chunk_count} 个父块，{result.child_chunk_count} 个子块。"
            )
            st.session_state.processed_uploads.add(fingerprint)
