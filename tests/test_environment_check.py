"""Environment diagnostics must be injectable and never expose credentials."""

from __future__ import annotations

from pathlib import Path

from rag_agent_platform.config import Settings
from scripts.check_environment import run_checks


class _Connection:
    def close(self) -> None:
        pass


def _settings(tmp_path: Path, **overrides: object) -> Settings:
    values: dict[str, object] = {
        "app_mode": "real",
        "document_repository_provider": "file",
        "embedding_provider": "hash",
        "metadata_path": str(tmp_path / "metadata/documents.json"),
        "chroma_persist_directory": str(tmp_path / "chroma"),
        "graph_persist_directory": str(tmp_path / "graph"),
        "upload_directory": str(tmp_path / "uploads"),
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


def test_environment_check_reports_safe_configured_status(tmp_path: Path) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text("LLM_API_KEY=super-secret-value\n", encoding="utf-8")
    settings = _settings(
        tmp_path,
        llm_provider="openai-compatible",
        llm_model="test-model",
        llm_api_key="super-secret-value",
        llm_base_url="https://example.invalid/v1",
    )

    results = run_checks(
        settings,
        env_path=env_path,
        python_version=(3, 12, 0),
        importer=lambda _name: object(),
        connector=lambda _address, timeout: _Connection(),
    )
    rendered = "\n".join(f"{item.name}: {item.detail}" for item in results)

    assert all(item.ok for item in results)
    assert "LLM credentials: configured" in rendered
    assert "super-secret-value" not in rendered


def test_environment_check_identifies_missing_provider_fields(tmp_path: Path) -> None:
    results = run_checks(
        _settings(tmp_path, llm_provider="openai-compatible"),
        env_path=tmp_path / ".env",
        python_version=(3, 12, 0),
        importer=lambda _name: object(),
        connector=lambda _address, timeout: _Connection(),
    )

    failed = {item.name: item.detail for item in results if not item.ok}
    assert ".env file" in failed
    assert "LLM configuration" in failed
    assert "LLM_MODEL" in failed["LLM configuration"]
    assert "LLM_API_KEY" in failed["LLM configuration"]


def test_mysql_connection_check_is_injectable(tmp_path: Path) -> None:
    calls: list[tuple[tuple[str, int], float]] = []

    def connector(address: tuple[str, int], timeout: float) -> _Connection:
        calls.append((address, timeout))
        return _Connection()

    results = run_checks(
        _settings(
            tmp_path,
            document_repository_provider="mysql",
            mysql_host="db.internal",
            mysql_port=3307,
            mysql_user="rag",
        ),
        env_path=tmp_path / ".env",
        python_version=(3, 12, 0),
        importer=lambda _name: object(),
        connector=connector,
    )

    mysql_result = next(item for item in results if item.name == "MySQL connectivity")
    assert mysql_result.ok is True
    assert calls == [(("db.internal", 3307), 5.0)]
