"""Executable package and dependency-direction rules.

These tests deliberately inspect imports with ``ast`` instead of matching source
substrings.  That keeps the rules lightweight while avoiding false positives in
docstrings and comments.
"""

from __future__ import annotations

import ast
from collections.abc import Iterable
from pathlib import Path

PACKAGE_ROOT = Path("src/rag_agent_platform")


def _python_files(directory: Path) -> Iterable[Path]:
    return (path for path in directory.rglob("*.py") if "__pycache__" not in path.parts)


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _imports(path: Path) -> set[str]:
    imported: set[str] = set()
    for node in ast.walk(_tree(path)):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    return imported


def _calls(path: Path) -> set[str]:
    called: set[str] = set()
    for node in ast.walk(_tree(path)):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name):
            called.add(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            called.add(node.func.attr)
    return called


def _assert_no_import_prefixes(path: Path, forbidden: tuple[str, ...]) -> None:
    violations = sorted(
        imported
        for imported in _imports(path)
        if any(imported == prefix or imported.startswith(f"{prefix}.") for prefix in forbidden)
    )
    assert not violations, f"{path}: forbidden imports {violations}"


def test_models_are_dependency_free_contracts() -> None:
    for path in _python_files(PACKAGE_ROOT / "models"):
        _assert_no_import_prefixes(
            path,
            (
                "rag_agent_platform.agent",
                "rag_agent_platform.evaluation",
                "rag_agent_platform.generation",
                "rag_agent_platform.graph",
                "rag_agent_platform.ingestion",
                "rag_agent_platform.retrieval",
                "rag_agent_platform.storage",
                "rag_agent_platform.ui",
                "dotenv",
                "pydantic_settings",
            ),
        )
        assert not _calls(path).intersection({"getenv", "load_dotenv"}), path


def test_ui_does_not_import_concrete_infrastructure() -> None:
    for path in _python_files(PACKAGE_ROOT / "ui"):
        _assert_no_import_prefixes(
            path,
            (
                "chromadb",
                "pymysql",
                "rag_agent_platform.storage.file",
                "rag_agent_platform.storage.mysql",
                "rag_agent_platform.retrieval.bm25",
                "rag_agent_platform.retrieval.dense",
                "rag_agent_platform.graph.store",
            ),
        )


def test_answer_layers_do_not_reach_sideways_into_retrieval_or_each_other() -> None:
    for path in _python_files(PACKAGE_ROOT / "generation"):
        _assert_no_import_prefixes(
            path,
            (
                "rag_agent_platform.agent",
                "rag_agent_platform.evaluation",
                "rag_agent_platform.retrieval",
                "rag_agent_platform.storage",
                "rag_agent_platform.ui",
            ),
        )

    _assert_no_import_prefixes(
        PACKAGE_ROOT / "evaluation" / "service.py",
        (
            "rag_agent_platform.agent",
            "rag_agent_platform.generation",
            "rag_agent_platform.retrieval",
            "rag_agent_platform.storage",
            "rag_agent_platform.ui",
        ),
    )


def test_agent_nodes_do_not_create_concrete_infrastructure() -> None:
    forbidden_imports = (
        "chromadb",
        "pymysql",
        "streamlit",
        "rag_agent_platform.storage.file",
        "rag_agent_platform.storage.mysql",
        "rag_agent_platform.graph.store",
    )
    forbidden_factories = {
        "ChromaVectorStore",
        "FileDocumentRepository",
        "MySQLDocumentRepository",
        "NetworkXGraphService",
        "OpenAICompatibleChatModel",
        "build_chat_model",
        "build_embedding_model",
    }
    for path in _python_files(PACKAGE_ROOT / "agent" / "nodes"):
        _assert_no_import_prefixes(path, forbidden_imports)
        violations = sorted(_calls(path).intersection(forbidden_factories))
        assert not violations, f"{path}: creates concrete services {violations}"


def test_interface_modules_do_not_import_concrete_implementations() -> None:
    base_modules = [
        PACKAGE_ROOT / package / "base.py"
        for package in (
            "agent",
            "embeddings",
            "evaluation",
            "generation",
            "graph",
            "ingestion",
            "retrieval",
            "storage",
        )
    ]
    for path in base_modules:
        rag_imports = {
            imported for imported in _imports(path) if imported.startswith("rag_agent_platform.")
        }
        violations = sorted(
            imported
            for imported in rag_imports
            if imported != "rag_agent_platform.models"
            and not imported.startswith("rag_agent_platform.models.")
        )
        assert not violations, f"{path}: base contract imports concrete layer {violations}"


def test_retrieval_and_storage_are_streamlit_independent() -> None:
    for package in ("retrieval", "storage"):
        for path in _python_files(PACKAGE_ROOT / package):
            _assert_no_import_prefixes(path, ("streamlit", "rag_agent_platform.ui"))


def test_ingestion_pipeline_does_not_select_a_concrete_repository() -> None:
    path = PACKAGE_ROOT / "ingestion" / "pipeline.py"
    _assert_no_import_prefixes(
        path,
        ("rag_agent_platform.storage.file", "rag_agent_platform.storage.mysql"),
    )


def test_project_uses_only_the_installed_package_import_path() -> None:
    roots = [Path("app.py"), PACKAGE_ROOT, Path("tests"), Path("scripts")]
    paths = [roots[0]]
    for root in roots[1:]:
        paths.extend(_python_files(root))

    for path in paths:
        if path.resolve() == Path(__file__).resolve():
            continue
        for node in ast.walk(_tree(path)):
            if isinstance(node, ast.Import):
                assert all(alias.name != "src" for alias in node.names), path
            elif isinstance(node, ast.ImportFrom):
                assert node.module != "src", path
            elif (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Attribute)
                and isinstance(node.func.value.value, ast.Name)
                and node.func.value.value.id == "sys"
                and node.func.value.attr == "path"
                and node.func.attr in {"append", "insert"}
            ):
                raise AssertionError(f"{path}: mutates sys.path")
