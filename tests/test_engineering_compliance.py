"""Small executable checks for engineering metadata and reproducible CI defaults."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import rag_agent_platform

ROOT = Path(__file__).parents[1]


def test_package_and_project_versions_match_and_are_documented() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    init_source = (ROOT / "src/rag_agent_platform/__init__.py").read_text(encoding="utf-8")
    package_version = re.search(r'__version__\s*=\s*"([^"]+)"', init_source)

    assert package_version is not None
    assert package_version.group(1) == project["project"]["version"]
    assert project["project"]["version"] == "1.0.0"
    assert rag_agent_platform.__version__ == "1.0.0"

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    release_notes = (ROOT / "docs/releases/v1.0.0.md").read_text(encoding="utf-8")
    acceptance = (ROOT / "docs/acceptance-report.md").read_text(encoding="utf-8")
    assert "Current stable version: `v1.0.0`" in readme
    assert "Previous milestone: `v0.7.0`" in readme
    assert "Project status: **Final course delivery**" in readme
    assert "[Unreleased]" in changelog
    assert "[1.0.0] - 2026-07-22" in changelog
    assert "[0.7.0] - 2026-07-21" in changelog
    assert "[0.1.0] - 2026-07-19" in changelog
    assert "v1.0.0" in release_notes
    assert "验收版本：`1.0.0`" in acceptance


def test_temporary_development_version_is_not_a_current_project_status() -> None:
    temporary_version = ".".join(("0", "8", "0")) + ".dev0"
    current_status_files = (
        "README.md",
        "CHANGELOG.md",
        "docs/acceptance-report.md",
        "docs/development-plan.md",
        "docs/engineering-checklist.md",
        "docs/engineering-practice.md",
        "docs/mvp-plan.md",
        "docs/release-process.md",
        "docs/requirements.md",
        "docs/releases/v1.0.0.md",
        "pyproject.toml",
        "src/rag_agent_platform/__init__.py",
    )
    combined = "\n".join(
        (ROOT / relative_path).read_text(encoding="utf-8") for relative_path in current_status_files
    )

    assert temporary_version not in combined
    assert "MySQL 预留" not in combined
    assert "GraphRAG 计划实现" not in combined
    assert "v1.0.0 尚未完成" not in combined
    assert "准备发布 v1.0.0" not in combined


def test_env_example_covers_settings_without_containing_a_secret() -> None:
    env_text = (ROOT / ".env.example").read_text(encoding="utf-8")
    keys = {
        line.partition("=")[0].strip()
        for line in env_text.splitlines()
        if line.strip() and not line.lstrip().startswith("#") and "=" in line
    }
    required = {
        "APP_MODE",
        "LLM_PROVIDER",
        "LLM_MODEL",
        "LLM_API_KEY",
        "EMBEDDING_PROVIDER",
        "EMBEDDING_API_KEY",
        "DOCUMENT_REPOSITORY_PROVIDER",
        "MYSQL_HOST",
        "MYSQL_PORT",
        "MYSQL_USER",
        "MYSQL_PASSWORD",
        "MYSQL_DATABASE",
        "CHROMA_PERSIST_DIRECTORY",
        "GRAPH_PERSIST_DIRECTORY",
        "UPLOAD_DIRECTORY",
        "AGENT_MAX_RETRIES",
        "AGENT_MAX_REGENERATIONS",
    }

    assert required <= keys
    assert not re.search(r"(?i)(?:sk-[A-Za-z0-9_-]{8,}|bearer\s+[A-Za-z0-9._-]{8,})", env_text)


def test_ci_has_reproducible_external_free_quality_gate() -> None:
    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")

    for required in (
        'python-version: "3.12"',
        "uv sync --frozen",
        "python -m compileall src tests scripts app.py",
        "ruff format --check .",
        "ruff check .",
        "pytest -q",
        "APP_MODE: mock",
        "EMBEDDING_PROVIDER: hash",
        "DOCUMENT_REPOSITORY_PROVIDER: file",
    ):
        assert required in workflow
    assert 'LLM_API_KEY: ""' in workflow
    assert "pull_request:" in workflow and "push:" in workflow


def test_numbered_requirements_are_present_in_traceability_matrix() -> None:
    requirements = (ROOT / "docs/requirements.md").read_text(encoding="utf-8")
    traceability = (ROOT / "docs/traceability.md").read_text(encoding="utf-8")
    requirement_ids = set(re.findall(r"\b(?:FR|NFR)-\d{2}\b", requirements))
    traced_ids = set(re.findall(r"\b(?:FR|NFR)-\d{2}\b", traceability))

    assert len({item for item in requirement_ids if item.startswith("FR-")}) >= 20
    assert len({item for item in requirement_ids if item.startswith("NFR-")}) >= 7
    assert requirement_ids <= traced_ids


def test_required_engineering_documents_exist() -> None:
    for relative_path in (
        "docs/testing-strategy.md",
        "docs/release-process.md",
        "docs/deployment.md",
        "docs/acceptance-report.md",
        "docs/demo-script.md",
        "docs/releases/v1.0.0.md",
        "scripts/check_environment.py",
    ):
        assert (ROOT / relative_path).is_file(), relative_path
