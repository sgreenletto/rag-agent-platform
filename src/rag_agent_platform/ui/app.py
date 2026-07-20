"""Streamlit application assembly."""

import streamlit as st

from rag_agent_platform.bootstrap import ServiceContainer, build_service_container
from rag_agent_platform.ui.components import (
    render_agent_result,
    render_chat_history,
    render_sidebar,
)
from rag_agent_platform.ui.session import initialize_session_state


@st.cache_resource
def build_services() -> ServiceContainer:
    """Create and cache heavyweight storage, embedding and graph resources."""
    return build_service_container()


def run_app() -> None:
    """Configure and render the complete Streamlit application."""
    st.set_page_config(page_title="模块化智能文档问答系统", page_icon="📄", layout="wide")
    initialize_session_state()
    try:
        container = build_services()
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        st.error(f"服务初始化失败：{exc}")
        st.info("请检查 .env 中的 APP_MODE、LLM、Embedding 与本地存储配置。")
        st.stop()

    selected_document_ids, current_mode = render_sidebar(
        container.ingestion,
        app_mode=container.app_mode,
        upload_directory=container.settings.upload_directory,
    )
    st.title(container.settings.app_name)
    st.subheader("模块化智能文档问答系统")
    if container.app_mode == "mock":
        st.warning("当前显式启用了 APP_MODE=mock；页面结果仅用于无外部依赖的开发测试。")
    elif not container.llm_configured:
        st.info(
            "当前未配置远程 LLM：系统仍使用真实入库与检索，并采用本地规则分类、"
            "保守抽取式生成和引用校验。配置 LLM_PROVIDER 后可启用模型生成与结构化评估。"
        )

    render_chat_history()
    query = st.chat_input("请输入你的问题")
    if not query:
        return

    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    try:
        result = container.agent.invoke(
            query=query,
            document_ids=selected_document_ids,
            mode=current_mode,
        )
    except (KeyError, TypeError, ValueError) as exc:
        st.error(f"问答执行失败：{exc}")
        return

    st.session_state.messages.append(
        {"role": "assistant", "content": result.answer, "result": result}
    )
    with st.chat_message("assistant"):
        render_agent_result(result)
