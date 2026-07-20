"""Document loader boundaries used by the real ingestion pipeline."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


@dataclass(slots=True)
class LoadedDocument:
    """Raw text extracted from one source document."""

    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class DocumentLoader(Protocol):
    """Load one local file into text and metadata."""

    supported_suffixes: set[str]

    def load(self, file_path: str | Path) -> LoadedDocument:
        """Return extracted text for one file."""
        ...


class LoaderRegistry:
    """Route file paths to suffix-aware document loaders."""

    def __init__(self, loaders: list[DocumentLoader] | None = None) -> None:
        self._loaders: dict[str, DocumentLoader] = {}
        for loader in loaders or []:
            self.register(loader)

    def register(self, loader: DocumentLoader) -> None:
        """Register one loader for all of its supported suffixes."""
        for suffix in loader.supported_suffixes:
            normalized_suffix = self._normalize_suffix(suffix)
            self._loaders[normalized_suffix] = loader

    def load(self, file_path: str | Path) -> LoadedDocument:
        """Load a file with the matching suffix-specific loader."""
        path = Path(file_path)
        loader = self._loaders.get(self._normalize_suffix(path.suffix))
        if loader is None:
            raise ValueError(f"unsupported file type: {path.suffix or '(none)'}")
        return loader.load(path)

    @staticmethod
    def _normalize_suffix(suffix: str) -> str:
        normalized = suffix.lower().strip()
        if normalized and not normalized.startswith("."):
            normalized = f".{normalized}"
        return normalized
