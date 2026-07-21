from pathlib import Path

PACKAGE_ROOT = Path("src/rag_agent_platform")


def _python_sources(directory: Path) -> dict[Path, str]:
    return {
        path: path.read_text(encoding="utf-8")
        for path in directory.rglob("*.py")
        if "__pycache__" not in path.parts
    }


def test_models_do_not_depend_on_upper_layers() -> None:
    forbidden = (
        "rag_agent_platform.agent",
        "rag_agent_platform.retrieval",
        "rag_agent_platform.ui",
        "rag_agent_platform.ingestion",
        "rag_agent_platform.storage",
    )

    for path, source in _python_sources(PACKAGE_ROOT / "models").items():
        assert not any(dependency in source for dependency in forbidden), path


def test_ui_does_not_import_infrastructure_or_retriever_implementations() -> None:
    forbidden = (
        "chromadb",
        "pymysql",
        "PersistentClient",
        "ChromaVectorStore",
        "HashEmbeddingModel",
        "BM25Retriever",
        "DenseRetriever",
        "NetworkXGraphService",
    )

    for path, source in _python_sources(PACKAGE_ROOT / "ui").items():
        assert not any(dependency in source for dependency in forbidden), path


def test_agent_nodes_do_not_create_infrastructure() -> None:
    forbidden = (
        "chromadb",
        "streamlit",
        "ChromaVectorStore",
        "PersistentClient",
        "NetworkXGraphService",
        "build_chat_model",
    )

    for path, source in _python_sources(PACKAGE_ROOT / "agent" / "nodes").items():
        assert not any(dependency in source for dependency in forbidden), path


def test_ingestion_pipeline_does_not_select_a_concrete_repository() -> None:
    source = (PACKAGE_ROOT / "ingestion" / "pipeline.py").read_text(encoding="utf-8")

    assert "FileDocumentRepository" not in source


def test_project_uses_only_the_installed_package_import_path() -> None:
    forbidden = ("from src", "import src", "sys.path.append", "sys.path.insert")
    roots = [Path("app.py"), PACKAGE_ROOT, Path("tests"), Path("scripts")]
    paths = [roots[0]]
    for root in roots[1:]:
        paths.extend(root.rglob("*.py"))

    for path in paths:
        if path.resolve() == Path(__file__).resolve():
            continue
        source = path.read_text(encoding="utf-8")
        assert not any(pattern in source for pattern in forbidden), path
