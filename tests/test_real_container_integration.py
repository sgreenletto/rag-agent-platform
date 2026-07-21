from pathlib import Path

import pytest

from rag_agent_platform.bootstrap import build_service_container
from rag_agent_platform.config import Settings
from rag_agent_platform.llm import build_chat_model
from rag_agent_platform.models import RetrievedChunk


def test_real_container_ingests_and_runs_all_manual_modes(tmp_path: Path) -> None:
    config = Settings(
        _env_file=None,
        app_mode="real",
        metadata_path=str(tmp_path / "metadata" / "documents.json"),
        chroma_persist_directory=str(tmp_path / "chroma"),
        chroma_collection_name="container_integration",
        graph_persist_directory=str(tmp_path / "graph"),
        upload_directory=str(tmp_path / "uploads"),
        agent_max_retries=0,
    )
    container = build_service_container(config)
    document_path = tmp_path / "purchase.txt"
    document_path.write_text(
        "采购部负责供应商管理。供应商影响原材料交付。员工每年享有十天年假。",
        encoding="utf-8",
    )
    ingestion_result = container.ingestion.ingest(document_path)
    document_ids = [ingestion_result.document.document_id]

    results = {
        "naive": container.agent.invoke("员工年假有几天？", document_ids, "naive"),
        "advanced": container.agent.invoke("综合说明员工年假制度", document_ids, "advanced"),
        "graph": container.agent.invoke("采购部 供应商 关系", document_ids, "graph"),
        "agent": container.agent.invoke("员工年假有几天？", document_ids, "agent"),
    }

    assert container.app_mode == "real"
    assert container.ingestion.list_documents() == [ingestion_result.document]
    assert all(result.error is None for result in results.values())
    assert all(
        isinstance(chunk, RetrievedChunk)
        for result in results.values()
        for chunk in result.retrieved_chunks
    )


def test_ui_modules_import_without_initializing_services() -> None:
    import rag_agent_platform.ui.app as ui_app
    import rag_agent_platform.ui.components as components

    assert callable(ui_app.run_app)
    assert callable(components.render_sidebar)


def test_configured_llm_reports_missing_credentials_clearly() -> None:
    config = Settings(_env_file=None, llm_provider="openai-compatible")

    try:
        build_chat_model(config)
    except ValueError as exc:
        assert "LLM_MODEL is required" in str(exc)
    else:
        raise AssertionError("missing configured LLM credentials must fail")


def test_real_container_reports_missing_embedding_credentials_clearly(tmp_path: Path) -> None:
    config = Settings(
        _env_file=None,
        app_mode="real",
        embedding_provider="siliconflow",
        metadata_path=str(tmp_path / "metadata" / "documents.json"),
        chroma_persist_directory=str(tmp_path / "chroma"),
        graph_persist_directory=str(tmp_path / "graph"),
    )

    try:
        build_service_container(config)
    except ValueError as exc:
        assert "EMBEDDING_MODEL is required" in str(exc)
    else:
        raise AssertionError("missing configured embedding credentials must fail")


def test_configuration_errors_do_not_expose_api_keys() -> None:
    secret = "test-secret-that-must-not-appear"
    config = Settings(
        _env_file=None,
        llm_provider="unsupported-provider",
        llm_api_key=secret,
    )

    with pytest.raises(ValueError) as error:
        build_chat_model(config)

    assert secret not in str(error.value)
