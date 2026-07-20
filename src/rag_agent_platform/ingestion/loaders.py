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


class TextDocumentLoader:
    """Load plain text and Markdown files with common local encodings."""

    supported_suffixes = {".txt", ".md"}
    _ENCODINGS = ("utf-8", "gbk", "gb18030", "utf-16")

    def load(self, file_path: str | Path) -> LoadedDocument:
        """Read one TXT or Markdown file."""
        path = Path(file_path)
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"document does not exist: {path}")
        if path.suffix.lower() not in self.supported_suffixes:
            raise ValueError(f"unsupported file type: {path.suffix or '(none)'}")

        content, encoding = self._read_text(path)
        if not content.strip():
            raise ValueError("document must not be empty")
        return LoadedDocument(
            content=content,
            metadata={
                "filename": path.name,
                "source_path": str(path.resolve()),
                "file_type": path.suffix.lower().lstrip("."),
                "encoding": encoding,
            },
        )

    def _read_text(self, path: Path) -> tuple[str, str]:
        for encoding in self._ENCODINGS:
            try:
                return path.read_text(encoding=encoding), encoding
            except UnicodeDecodeError:
                continue
        raise UnicodeDecodeError(
            "text",
            b"",
            0,
            1,
            f"unable to decode document with supported encodings: {', '.join(self._ENCODINGS)}",
        )


def build_default_loader_registry() -> LoaderRegistry:
    """Build the default loader registry used by demos and the real pipeline."""
    return LoaderRegistry([TextDocumentLoader()])
