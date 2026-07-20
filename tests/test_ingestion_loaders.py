from pathlib import Path

import pytest

from rag_agent_platform.ingestion.loaders import (
    LoaderRegistry,
    TextDocumentLoader,
    build_default_loader_registry,
)


@pytest.mark.parametrize(
    ("filename", "content"),
    [
        ("leave_policy.txt", "员工请假制度\n年假需要提前申请。"),
        ("purchase_process.md", "# 采购流程\n采购申请需要部门负责人审批。"),
    ],
)
def test_text_and_markdown_can_be_loaded(
    tmp_path: Path,
    filename: str,
    content: str,
) -> None:
    file_path = tmp_path / filename
    file_path.write_text(content, encoding="utf-8")

    loaded = build_default_loader_registry().load(file_path)

    assert loaded.content == content
    assert loaded.metadata["filename"] == filename
    assert loaded.metadata["file_type"] == file_path.suffix.lower().lstrip(".")
    assert loaded.metadata["encoding"] == "utf-8"
    assert loaded.metadata["source_path"].endswith(filename)


def test_gbk_text_can_be_loaded(tmp_path: Path) -> None:
    file_path = tmp_path / "gbk-policy.txt"
    content = "财务报销制度"
    file_path.write_text(content, encoding="gbk")

    loaded = build_default_loader_registry().load(file_path)

    assert loaded.content == content
    assert loaded.metadata["encoding"] in {"gbk", "gb18030"}


def test_empty_file_fails(tmp_path: Path) -> None:
    file_path = tmp_path / "empty.txt"
    file_path.write_text("  \n", encoding="utf-8")

    with pytest.raises(ValueError, match="must not be empty"):
        build_default_loader_registry().load(file_path)


def test_missing_file_fails(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="does not exist"):
        build_default_loader_registry().load(tmp_path / "missing.txt")


def test_unsupported_file_type_fails(tmp_path: Path) -> None:
    file_path = tmp_path / "image.png"
    file_path.write_bytes(b"not a document")

    with pytest.raises(ValueError, match="unsupported file type"):
        build_default_loader_registry().load(file_path)


def test_loader_registry_routes_registered_suffix(tmp_path: Path) -> None:
    file_path = tmp_path / "sample.txt"
    file_path.write_text("示例文本", encoding="utf-8")
    registry = LoaderRegistry([TextDocumentLoader()])

    loaded = registry.load(file_path)

    assert loaded.metadata["filename"] == "sample.txt"
