"""Streamlit smoke test using the explicit external-free mock mode."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

import rag_agent_platform.bootstrap as bootstrap
from rag_agent_platform.config import Settings
from rag_agent_platform.ui.app import build_services


def test_streamlit_app_initializes_without_uncaught_exception(
    monkeypatch,
    tmp_path: Path,
) -> None:
    config = Settings(
        _env_file=None,
        app_mode="mock",
        metadata_path=str(tmp_path / "metadata/documents.json"),
        chroma_persist_directory=str(tmp_path / "chroma"),
        graph_persist_directory=str(tmp_path / "graph"),
        upload_directory=str(tmp_path / "uploads"),
    )
    monkeypatch.setattr(bootstrap, "settings", config)
    build_services.clear()

    app = AppTest.from_file("app.py").run(timeout=20)

    assert not app.exception
    assert app.title[0].value == "RAG Agent Platform"
    assert any("APP_MODE=mock" in warning.value for warning in app.warning)
    build_services.clear()
