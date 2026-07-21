from pathlib import Path

import pytest

from rag_agent_platform.config import Settings
from rag_agent_platform.storage import FileDocumentRepository, build_document_repository


def test_build_document_repository_defaults_to_file(tmp_path: Path) -> None:
    settings = Settings(
        _env_file=None,
        document_repository_provider="",
        metadata_path=str(tmp_path / "documents.json"),
    )

    repository = build_document_repository(settings)

    assert isinstance(repository, FileDocumentRepository)


def test_build_document_repository_builds_mysql(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class FakeMySQLDocumentRepository:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(
        "rag_agent_platform.storage.factory.MySQLDocumentRepository",
        FakeMySQLDocumentRepository,
    )
    settings = Settings(
        _env_file=None,
        document_repository_provider="mysql",
        mysql_host="db.local",
        mysql_port=3307,
        mysql_user="rag_user",
        mysql_password="secret",
        mysql_database="rag_agent_test",
        mysql_charset="utf8mb4",
        mysql_connect_timeout=3,
    )

    repository = build_document_repository(settings)

    assert isinstance(repository, FakeMySQLDocumentRepository)
    assert captured == {
        "host": "db.local",
        "port": 3307,
        "user": "rag_user",
        "password": "secret",
        "database": "rag_agent_test",
        "charset": "utf8mb4",
        "connect_timeout": 3,
    }


def test_build_document_repository_rejects_unknown_provider() -> None:
    settings = Settings(_env_file=None, document_repository_provider="postgres")

    with pytest.raises(ValueError, match="DOCUMENT_REPOSITORY_PROVIDER"):
        build_document_repository(settings)
