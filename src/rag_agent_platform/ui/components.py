"""Reusable Streamlit presentation components."""

import hashlib
from pathlib import Path
from typing import Any

import streamlit as st

from rag_agent_platform.ingestion.base import IngestionPipeline
from rag_agent_platform.models import AgentResult


def render_sidebar(
    ingestion: IngestionPipeline,
    *,
    app_mode: str,
    upload_directory: str,
) -> tuple[list[str], str]:
    """Render real upload, document management and mode controls."""
    with st.sidebar:
        st.title("RAG Agent Platform")
        st.caption(f"运行模式：{app_mode}")
        supported_types = ["txt", "md"] if app_mode == "mock" else ["txt", "md", "pdf", "docx"]
        uploads = st.file_uploader(
            "上传文档",
            type=supported_types,
            accept_multiple_files=True,
            help=f"当前支持：{', '.join(supported_types)}",
        )
        _process_uploads(uploads, ingestion, Path(upload_directory))

        documents = ingestion.list_documents()
        st.subheader("已有文档")
        if not documents:
            st.caption("暂无文档")
        for document in documents:
            name_column, delete_column = st.columns([4, 1])
            name_column.caption(f"{document.filename} · {document.status}")
            if delete_column.button("删除", key=f"delete-{document.document_id}"):
                _delete_document(document.document_id, ingestion)

        document_names = {document.document_id: document.filename for document in documents}
        st.session_state.selected_document_ids = [
            document_id
            for document_id in st.session_state.selected_document_ids
            if document_id in document_names
        ]
        selected_document_ids = st.multiselect(
            "选择知识源",
            options=list(document_names),
            format_func=lambda document_id: document_names[document_id],
            key="selected_document_ids",
            help="不选择文档时检索范围为空；可多选限定知识库范围。",
        )

        mode_labels = {
            "agent": "Agent 自动模式",
            "naive": "Naive RAG",
            "advanced": "Advanced RAG",
            "graph": "GraphRAG",
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
    """Render the answer, diagnostics, evidence methods and citations."""
    if result.error:
        st.error(result.error)
    st.markdown(result.answer)
    with st.expander("Agent 执行信息与轨迹"):
        st.write(f"query_type：`{result.query_type.value}`")
        st.write(f"strategy：`{result.strategy.value}`")
        st.write(f"retry_count：`{result.retry_count}`")
        methods = sorted({chunk.retrieval_method for chunk in result.retrieved_chunks})
        st.write(f"retrieval_method：`{', '.join(methods) if methods else 'none'}`")
        for step in result.execution_trace:
            st.write(f"- {step}")

    with st.expander("来源与引用"):
        if not result.citations:
            st.caption("本次回答没有知识库引用。")
        for citation in result.citations:
            page = f"，第 {citation.page} 页" if citation.page is not None else ""
            chunk = f"，chunk={citation.chunk_id}" if citation.chunk_id else ""
            st.text(f"[{citation.index}] {citation.source}{page}{chunk}")


def _process_uploads(
    uploads: list[Any] | None,
    ingestion: IngestionPipeline,
    upload_directory: Path,
) -> None:
    if not uploads:
        return
    upload_directory.mkdir(parents=True, exist_ok=True)
    for upload in uploads:
        content = upload.getvalue()
        fingerprint = hashlib.sha256(upload.name.encode("utf-8") + content).hexdigest()
        if fingerprint in st.session_state.processed_uploads:
            continue
        safe_name = Path(upload.name).name
        target = upload_directory / f"{fingerprint[:12]}-{safe_name}"
        try:
            target.write_bytes(content)
            result = ingestion.ingest(target)
        except (OSError, RuntimeError, UnicodeDecodeError, ValueError) as exc:
            st.error(f"{upload.name} 入库失败：{exc}")
        else:
            st.success(
                f"{upload.name} 已完成入库：{result.parent_chunk_count} 个父块，"
                f"{result.child_chunk_count} 个子块。"
            )
            st.session_state.processed_uploads.add(fingerprint)
            if result.document.document_id not in st.session_state.selected_document_ids:
                st.session_state.selected_document_ids.append(result.document.document_id)


def _delete_document(document_id: str, ingestion: IngestionPipeline) -> None:
    try:
        ingestion.delete_document(document_id)
    except (OSError, RuntimeError, ValueError) as exc:
        st.error(f"删除文档失败：{exc}")
        return
    st.session_state.selected_document_ids = [
        selected for selected in st.session_state.selected_document_ids if selected != document_id
    ]
    st.success("文档及其向量、稀疏索引和图数据已删除。")
    st.rerun()
