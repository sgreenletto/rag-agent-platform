"""Validate local prerequisites without printing credentials.

Run with ``uv run python scripts/check_environment.py``.  The checks are kept
injectable so they can be tested without opening real sockets or importing
external services.
"""

from __future__ import annotations

import importlib
import socket
import sys
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rag_agent_platform.config import Settings


@dataclass(frozen=True, slots=True)
class CheckResult:
    """One safe environment diagnostic."""

    name: str
    ok: bool
    detail: str


def run_checks(
    settings: Settings,
    *,
    env_path: Path = Path(".env"),
    python_version: tuple[int, int, int] | tuple[int, int] = sys.version_info[:3],
    importer: Callable[[str], Any] = importlib.import_module,
    connector: Callable[[tuple[str, int], float], Any] = socket.create_connection,
) -> list[CheckResult]:
    """Return diagnostics for the configured local application environment."""
    results = [
        CheckResult(
            "Python version",
            python_version >= (3, 12),
            f"{python_version[0]}.{python_version[1]} (requires >= 3.12)",
        ),
        CheckResult(
            ".env file",
            env_path.is_file(),
            "present" if env_path.is_file() else "missing; copy .env.example to .env",
        ),
    ]
    results.extend(_configuration_checks(settings))
    results.extend(_directory_checks(settings))
    results.extend(_import_checks(importer))
    if settings.document_repository_provider.strip().lower() == "mysql":
        results.append(_mysql_check(settings, connector))
    return results


def _configuration_checks(settings: Settings) -> list[CheckResult]:
    results: list[CheckResult] = []
    app_mode = settings.app_mode.strip().lower()
    results.append(
        CheckResult(
            "Application mode",
            app_mode in {"real", "mock"},
            app_mode if app_mode in {"real", "mock"} else "must be real or mock",
        )
    )

    llm_provider = settings.llm_provider.strip().lower()
    if not llm_provider:
        results.append(CheckResult("LLM configuration", True, "optional provider disabled"))
        results.append(CheckResult("LLM credentials", True, "missing (not required)"))
    else:
        missing = [
            name
            for name, value in (
                ("LLM_MODEL", settings.llm_model),
                ("LLM_API_KEY", settings.llm_api_key),
            )
            if not value.strip()
        ]
        results.append(
            CheckResult(
                "LLM configuration",
                not missing,
                "complete" if not missing else f"missing: {', '.join(missing)}",
            )
        )
        results.append(
            CheckResult(
                "LLM credentials",
                bool(settings.llm_api_key.strip()),
                "configured" if settings.llm_api_key.strip() else "missing",
            )
        )

    embedding_provider = settings.embedding_provider.strip().lower()
    if embedding_provider in {"", "hash", "local"}:
        results.append(CheckResult("Embedding configuration", True, "local/hash mode"))
    else:
        missing = [
            name
            for name, value in (
                ("EMBEDDING_MODEL", settings.embedding_model),
                ("EMBEDDING_API_KEY", settings.embedding_api_key),
            )
            if not value.strip()
        ]
        results.append(
            CheckResult(
                "Embedding configuration",
                not missing,
                "complete" if not missing else f"missing: {', '.join(missing)}",
            )
        )

    repository = settings.document_repository_provider.strip().lower()
    repository_ok = repository in {"file", "mysql"}
    results.append(
        CheckResult(
            "Repository provider",
            repository_ok,
            repository if repository_ok else "must be file or mysql",
        )
    )
    if repository == "mysql":
        missing = [
            name
            for name, value in (
                ("MYSQL_HOST", settings.mysql_host),
                ("MYSQL_USER", settings.mysql_user),
                ("MYSQL_DATABASE", settings.mysql_database),
            )
            if not str(value).strip()
        ]
        results.append(
            CheckResult(
                "MySQL configuration",
                not missing,
                "complete" if not missing else f"missing: {', '.join(missing)}",
            )
        )
    return results


def _directory_checks(settings: Settings) -> list[CheckResult]:
    directories = {
        "metadata": Path(settings.metadata_path).parent,
        "Chroma": Path(settings.chroma_persist_directory),
        "graph": Path(settings.graph_persist_directory),
        "uploads": Path(settings.upload_directory),
    }
    results: list[CheckResult] = []
    for label, directory in directories.items():
        try:
            directory.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=directory):
                pass
        except OSError as exc:
            results.append(
                CheckResult(
                    f"{label} directory",
                    False,
                    f"not writable: {type(exc).__name__}",
                )
            )
        else:
            results.append(CheckResult(f"{label} directory", True, "writable"))
    return results


def _import_checks(importer: Callable[[str], Any]) -> list[CheckResult]:
    packages = (
        "rag_agent_platform",
        "chromadb",
        "jieba",
        "langgraph",
        "networkx",
        "pymysql",
        "streamlit",
    )
    results: list[CheckResult] = []
    for package in packages:
        try:
            importer(package)
        except (ImportError, ModuleNotFoundError) as exc:
            results.append(
                CheckResult(package, False, f"import failed: {type(exc).__name__}; run uv sync")
            )
        else:
            results.append(CheckResult(package, True, "importable"))
    return results


def _mysql_check(
    settings: Settings,
    connector: Callable[[tuple[str, int], float], Any],
) -> CheckResult:
    try:
        connection = connector(
            (settings.mysql_host, settings.mysql_port),
            float(settings.mysql_connect_timeout),
        )
        connection.close()
    except OSError as exc:
        return CheckResult(
            "MySQL connectivity",
            False,
            f"unreachable: {type(exc).__name__}; check host, port and service state",
        )
    return CheckResult("MySQL connectivity", True, "TCP endpoint reachable")


def main() -> int:
    """Print safe diagnostics and return a shell-friendly status code."""
    results = run_checks(Settings())
    for result in results:
        state = "PASS" if result.ok else "FAIL"
        print(f"[{state}] {result.name}: {result.detail}")
    if all(result.ok for result in results):
        print("Environment check passed.")
        return 0
    print("Environment check failed; resolve the FAIL items and run again.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
