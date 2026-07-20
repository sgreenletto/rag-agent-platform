"""NetworkX-backed GraphService implementation.

Orchestrates a :class:`TripletExtractor` and
:class:`NetworkXGraphStore` to fulfil the ``GraphService`` contract.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rag_agent_platform.graph.base import GraphService
from rag_agent_platform.graph.extractor import TripletExtractor
from rag_agent_platform.graph.models import Triplet
from rag_agent_platform.graph.store import NetworkXGraphStore

# Imported from the public models package (created by member 2).
# We use duck-typing fallbacks so the module is importable even
# when the models package hasn't been merged yet.
try:
    from rag_agent_platform.models import ChildChunk, RetrievedChunk
except ImportError:  # pragma: no cover — models arrive on merge
    ChildChunk = None  # type: ignore[assignment]
    RetrievedChunk = None  # type: ignore[assignment]


class NetworkXGraphService(GraphService):
    """GraphRAG service backed by NetworkX + rule-based extraction.

    Parameters:
        extractor: A :class:`TripletExtractor` implementation.
        store: Optional pre-configured :class:`NetworkXGraphStore`.
            When *None* an empty store is created.
        persist_dir: When set, the graph is auto-saved to
            ``<persist_dir>/graph.json`` after :meth:`build` and can
            be loaded with :meth:`load`.
        document_sources: Optional ``{document_id: source_filename}``
            mapping used to populate ``RetrievedChunk.source``.
    """

    _DEFAULT_SOURCE = "graph-evidence"

    def __init__(
        self,
        extractor: TripletExtractor,
        store: NetworkXGraphStore | None = None,
        persist_dir: Path | None = None,
        document_sources: dict[str, str] | None = None,
    ) -> None:
        self._extractor = extractor
        self._store = store if store is not None else NetworkXGraphStore()
        self._persist_dir = Path(persist_dir) if persist_dir else None

        # Fast lookup tables populated during build()
        self._chunk_content: dict[str, str] = {}
        self._chunk_parent: dict[str, str | None] = {}
        self._chunk_page: dict[str, int | None] = {}
        self._chunk_document: dict[str, str] = {}
        self._document_sources: dict[str, str] = dict(document_sources) if document_sources else {}

    # ------------------------------------------------------------------
    # GraphService contract
    # ------------------------------------------------------------------

    def build(self, chunks: list[object]) -> None:
        """Build / rebuild the knowledge graph from *chunks*.

        Args:
            chunks: ``list[ChildChunk]``.
        """
        if not chunks:
            return

        # 1. Index chunk metadata for later retrieval lookups
        for chunk in chunks:
            cid: str = getattr(chunk, "chunk_id", "")
            if not cid:
                continue
            self._chunk_content[cid] = getattr(chunk, "content", "")
            self._chunk_parent[cid] = getattr(chunk, "parent_id", None)
            self._chunk_page[cid] = getattr(chunk, "page", None)
            doc_id: str = getattr(chunk, "document_id", "")
            self._chunk_document[cid] = doc_id
            # Auto-register source from chunk metadata or document_id
            if doc_id and doc_id not in self._document_sources:
                meta: dict[str, Any] = getattr(chunk, "metadata", {}) or {}
                source = meta.get("source", "") or doc_id
                self._document_sources[doc_id] = source

        # 2. Extract triplets
        triplets = self._extractor.extract(chunks)

        # 3. Populate graph store
        self._store.add_triplets(triplets)

        # 4. Auto-persist
        if self._persist_dir:
            self._store.to_json(self._persist_dir / "graph.json")

    def delete_document(self, document_id: str) -> None:
        """Remove all graph data belonging to *document_id*."""
        if not document_id:
            return

        # Purge chunk index entries for this document
        stale_chunks = [cid for cid, doc in self._chunk_document.items() if doc == document_id]
        for cid in stale_chunks:
            self._chunk_content.pop(cid, None)
            self._chunk_parent.pop(cid, None)
            self._chunk_page.pop(cid, None)
            self._chunk_document.pop(cid, None)

        # Purge document source
        self._document_sources.pop(document_id, None)

        # Purge graph
        self._store.remove_document(document_id)

        # Re-persist
        if self._persist_dir:
            self._store.to_json(self._persist_dir / "graph.json")

    def retrieve(
        self,
        query: str,
        document_ids: list[str] | None = None,
        top_k: int = 5,
    ) -> list[RetrievedChunk]:
        """Return graph evidence as ``list[RetrievedChunk]``.

        1. Extract entity mentions from *query* via the extractor.
        2. Search the graph neighbourhood (1-2 hops).
        3. Map matched chunk ids back to ``RetrievedChunk`` objects
           sorted by ``normalized_score`` descending.
        """
        if not query or not query.strip():
            raise ValueError("query must not be empty")
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        # 1. Extract entities from the query text
        query_triplets: list[Triplet] = self._extractor.extract_from_text(query)
        entities: list[str] = []
        seen_entities: set[str] = set()
        for t in query_triplets:
            for entity in (t.subject, t.object):
                key = entity.strip().lower()
                if key and key not in seen_entities:
                    seen_entities.add(key)
                    entities.append(entity)

        # If no entities were extracted, fall back to keyword tokens
        if not entities:
            entities = _tokenize_query(query)

        # 2. Search the graph
        chunk_scores = self._store.search_neighborhood(
            entities=entities,
            document_ids=document_ids,
            top_k=top_k,
        )

        # 3. Assemble RetrievedChunk objects
        results: list[RetrievedChunk] = []
        for chunk_id, score in chunk_scores:
            content = self._chunk_content.get(chunk_id, "")
            if not content:
                continue
            doc_id = self._chunk_document.get(chunk_id, "")
            source = self._document_sources.get(doc_id, doc_id or self._DEFAULT_SOURCE)
            parent_id = self._chunk_parent.get(chunk_id)
            page = self._chunk_page.get(chunk_id)

            try:
                chunk = RetrievedChunk(
                    chunk_id=chunk_id,
                    content=content,
                    normalized_score=round(score, 4),
                    source=source,
                    document_id=doc_id or None,
                    parent_id=parent_id,
                    page=page,
                    retrieval_method="graph",
                    metadata={
                        "query_entities": entities,
                        "graph_score": round(score, 4),
                    },
                )
                results.append(chunk)
            except ValueError:
                # Invalid legacy chunks cannot satisfy the public RetrievedChunk contract.
                continue

        # Results are already sorted by the store; ensure descending
        results.sort(key=lambda c: getattr(c, "normalized_score", 0.0), reverse=True)
        return results

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def load(self, path: Path | None = None) -> None:
        """Load a previously persisted graph JSON file.

        If *path* is *None* and *persist_dir* was set at init, loads
        ``<persist_dir>/graph.json``.
        """
        resolved = Path(path) if path else None
        if resolved is None:
            if self._persist_dir is None:
                raise ValueError("no path given and persist_dir was not configured")
            resolved = self._persist_dir / "graph.json"
        self._store = NetworkXGraphStore.from_json(resolved)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def store(self) -> NetworkXGraphStore:
        """Expose the underlying graph store (read-only intended)."""
        return self._store

    @property
    def chunk_count(self) -> int:
        """Number of indexed chunks."""
        return len(self._chunk_content)

    @property
    def entity_count(self) -> int:
        """Number of entity nodes in the graph."""
        return len(self._store)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tokenize_query(query: str) -> list[str]:
    """Fallback: tokenize a query into potential entity tokens.

    Used when the extractor finds no structured entities in the query.
    """
    # Remove punctuation and split on whitespace
    cleaned = query.strip()
    # Simple Chinese + English tokenisation
    tokens: list[str] = []
    # Extract contiguous word sequences (Chinese chars or Latin words)
    import re

    # Match CJK sequences or Latin/num sequences
    for match in re.finditer(r"[一-鿿㐀-䶿]{1,8}|[A-Za-z0-9_]{2,}", cleaned):
        token = match.group(0).strip()
        if token and len(token) >= 2:
            tokens.append(token)

    if not tokens:
        # Last resort: split by whitespace
        tokens = [t for t in cleaned.split() if len(t) >= 2]

    return tokens[:10]  # Cap to avoid flooding the graph search
