"""NetworkX-based knowledge graph store with JSON persistence.

Provides deterministic, in-memory graph operations backed by
``networkx.DiGraph``.  Supports incremental triplet insertion with
entity normalisation, document-scoped deletion, BFS neighbourhood
search, and round-trip JSON serialisation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import networkx as nx

from rag_agent_platform.graph.models import Triplet


class NetworkXGraphStore:
    """Directed knowledge graph backed by NetworkX.

    Nodes represent entities (canonicalised lowercase names); edges
    represent relations with provenance metadata.

    Parameters:
        graph: Optionally seed with an existing ``nx.DiGraph``.
    """

    # Default weight applied per evidence chunk
    _DEFAULT_EDGE_WEIGHT = 1.0

    def __init__(self, graph: nx.DiGraph | None = None) -> None:
        self._graph = graph if graph is not None else nx.DiGraph()

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add_triplets(self, triplets: list[Triplet]) -> None:
        """Insert *triplets* into the graph, merging duplicate entities.

        Nodes are canonicalised by lowercased entity text.  When a node
        or edge already exists its ``chunk_ids`` and ``document_ids``
        sets are extended and the edge weight is incremented.
        """
        if not triplets:
            return

        for t in triplets:
            subj_key = _entity_key(t.subject)
            obj_key = _entity_key(t.object)

            # Ensure nodes exist
            for key, label in ((subj_key, t.subject), (obj_key, t.object)):
                if key not in self._graph:
                    self._graph.add_node(
                        key,
                        label=label,
                        type="entity",
                        chunk_ids=set(),
                        document_ids=set(),
                    )

            # Register chunk provenance on nodes (skip empty chunk_ids)
            chunk_id = t.source_chunk_id
            doc_id = _infer_document_id(chunk_id)
            if chunk_id:
                for key in (subj_key, obj_key):
                    node = self._graph.nodes[key]
                    node["chunk_ids"].add(chunk_id)
                    if doc_id:
                        node["document_ids"].add(doc_id)

            # Upsert edge (skip edges with no chunk provenance)
            if chunk_id:
                if self._graph.has_edge(subj_key, obj_key):
                    edge_data = self._graph.edges[subj_key, obj_key]
                    # Only update weight if it's the same predicate
                    if edge_data.get("predicate", "").lower() == t.predicate.lower():
                        edge_data["weight"] = (
                            edge_data.get("weight", 0.0) + self._DEFAULT_EDGE_WEIGHT
                        )
                        edge_data["chunk_ids"].add(chunk_id)
                        if doc_id:
                            edge_data.setdefault("document_ids", set()).add(doc_id)
                else:
                    self._graph.add_edge(
                        subj_key,
                        obj_key,
                        predicate=t.predicate,
                        weight=self._DEFAULT_EDGE_WEIGHT,
                        chunk_ids={chunk_id},
                        document_ids={doc_id} if doc_id else set(),
                    )

    def remove_document(self, document_id: str) -> None:
        """Purge all nodes and edges exclusively owned by *document_id*.

        Nodes that are shared with other documents are kept; only the
        document's chunk_ids are removed from their provenance sets.
        """
        if not document_id:
            return

        # Collect chunk_ids owned by this document
        doc_chunk_ids: set[str] = set()
        nodes_to_remove: list[str] = []
        for node_id, data in self._graph.nodes(data=True):
            data["chunk_ids"] = {
                cid
                for cid in data.get("chunk_ids", set())
                if _infer_document_id(cid) != document_id
            }
            data["document_ids"].discard(document_id)
            if not data["chunk_ids"]:
                nodes_to_remove.append(node_id)
            else:
                doc_chunk_ids.update(
                    cid
                    for cid in data.get("chunk_ids", set())
                    if _infer_document_id(cid) == document_id
                )

        # Remove edges that reference any of this document's chunk_ids
        edges_to_remove: list[tuple[str, str]] = []
        for u, v, data in self._graph.edges(data=True):
            data["chunk_ids"] = {
                cid
                for cid in data.get("chunk_ids", set())
                if _infer_document_id(cid) != document_id
            }
            data.setdefault("document_ids", set()).discard(document_id)
            if not data["chunk_ids"]:
                edges_to_remove.append((u, v))

        self._graph.remove_edges_from(edges_to_remove)
        self._graph.remove_nodes_from(
            [
                n
                for n in nodes_to_remove
                if n not in self._graph or not self._graph.nodes[n].get("chunk_ids")
            ]
        )

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def search_neighborhood(
        self,
        entities: list[str],
        document_ids: list[str] | None = None,
        top_k: int = 5,
    ) -> list[tuple[str, float]]:
        """Return relevant ``(chunk_id, score)`` pairs for *entities*.

        For each entity the graph performs a 1-2 hop BFS from the
        matching node(s).  Chunks attached to edges traversed during
        the walk are scored higher when the path is shorter and carries
        more evidence edges.

        Args:
            entities: Entity name strings to anchor the search.
            document_ids: Optional allow-list of document ids.
            top_k: Maximum number of chunk results.

        Returns:
            Sorted list of ``(chunk_id, score)``, ordered by score
            descending.  Scores are in ``[0.0, 1.0]``.
        """
        if not entities or top_k <= 0:
            return []

        allow_docs: set[str] | None = set(document_ids) if document_ids else None

        # Find matching nodes for each entity
        seed_nodes: list[str] = []
        for entity in entities:
            key = _entity_key(entity)
            if key in self._graph:
                seed_nodes.append(key)
            else:
                # Substring / partial match fallback
                for node_id in self._graph.nodes():
                    if key in node_id or node_id in key:
                        seed_nodes.append(node_id)

        if not seed_nodes:
            return []

        # BFS from seed nodes, collect (chunk_id, distance, edge_weight)
        chunk_scores: dict[str, float] = {}
        visited: set[str] = set()

        for seed in seed_nodes:
            # Level 0 (the node itself): collect chunk_ids from the node
            if seed in self._graph:
                node_data = self._graph.nodes[seed]
                for cid in node_data.get("chunk_ids", set()):
                    if _chunk_allowed(cid, allow_docs):
                        chunk_scores[cid] = max(chunk_scores.get(cid, 0.0), 1.0)

            # BFS queue: (node, distance)
            queue: list[tuple[str, int]] = [(seed, 0)]
            visited.add(seed)

            while queue:
                current, dist = queue.pop(0)
                if dist >= 2:  # Only expand up to 2 hops
                    continue

                # Outgoing edges
                for _, neighbor, edge_data in self._graph.out_edges(current, data=True):
                    self._score_edge_chunks(edge_data, dist + 1, allow_docs, chunk_scores)
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, dist + 1))

                # Incoming edges
                for neighbor, _, edge_data in self._graph.in_edges(current, data=True):
                    self._score_edge_chunks(edge_data, dist + 1, allow_docs, chunk_scores)
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, dist + 1))

        # Normalise scores to [0, 1]
        if chunk_scores:
            max_score = max(chunk_scores.values())
            if max_score > 0:
                chunk_scores = {k: v / max_score for k, v in chunk_scores.items()}

        # Sort and truncate
        ranked = sorted(chunk_scores.items(), key=lambda x: (-x[1], x[0]))
        return ranked[:top_k]

    @staticmethod
    def _score_edge_chunks(
        edge_data: dict[str, Any],
        distance: int,
        allow_docs: set[str] | None,
        chunk_scores: dict[str, float],
    ) -> None:
        """Accumulate scores for chunks referenced by an edge."""
        weight = edge_data.get("weight", 1.0)
        # Score formula: closer edges + higher weight → higher score
        base = weight / float(distance)
        for cid in edge_data.get("chunk_ids", set()):
            if not _chunk_allowed(cid, allow_docs):
                continue
            chunk_scores[cid] = max(chunk_scores.get(cid, 0.0), base)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def to_json(self, path: Path) -> None:
        """Serialize the graph to a JSON file.

        The output format is a dict with ``nodes``, ``edges``, and
        ``metadata`` keys.  ``set`` values are converted to lists.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        nodes = []
        for node_id, data in self._graph.nodes(data=True):
            nodes.append(
                {
                    "id": node_id,
                    "label": data.get("label", node_id),
                    "type": data.get("type", "entity"),
                    "chunk_ids": sorted(data.get("chunk_ids", set())),
                    "document_ids": sorted(data.get("document_ids", set())),
                }
            )

        edges = []
        for u, v, data in self._graph.edges(data=True):
            edges.append(
                {
                    "source": u,
                    "target": v,
                    "predicate": data.get("predicate", ""),
                    "weight": data.get("weight", 1.0),
                    "chunk_ids": sorted(data.get("chunk_ids", set())),
                    "document_ids": sorted(data.get("document_ids", set())),
                }
            )

        payload: dict[str, Any] = {
            "nodes": nodes,
            "edges": edges,
            "metadata": {
                "entity_count": len(nodes),
                "edge_count": len(edges),
                "triplet_count": sum(len(e.get("chunk_ids", [])) for e in edges),
            },
        }

        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
            encoding="utf-8",
        )

    @classmethod
    def from_json(cls, path: Path) -> NetworkXGraphStore:
        """Deserialize a graph that was persisted with :meth:`to_json`."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"graph JSON file not found: {path}")

        data = json.loads(path.read_text(encoding="utf-8"))

        graph = nx.DiGraph()
        for node in data.get("nodes", []):
            graph.add_node(
                node["id"],
                label=node.get("label", node["id"]),
                type=node.get("type", "entity"),
                chunk_ids=set(node.get("chunk_ids", [])),
                document_ids=set(node.get("document_ids", [])),
            )
        for edge in data.get("edges", []):
            graph.add_edge(
                edge["source"],
                edge["target"],
                predicate=edge.get("predicate", ""),
                weight=edge.get("weight", 1.0),
                chunk_ids=set(edge.get("chunk_ids", [])),
                document_ids=set(edge.get("document_ids", [])),
            )

        return cls(graph=graph)

    # ------------------------------------------------------------------
    # Introspection (useful for testing)
    # ------------------------------------------------------------------

    @property
    def graph(self) -> nx.DiGraph:
        """Expose the underlying NetworkX graph (read-only intended)."""
        return self._graph

    def clear(self) -> None:
        """Remove all nodes and edges."""
        self._graph.clear()

    def __len__(self) -> int:
        return self._graph.number_of_nodes()

    def __contains__(self, entity: str) -> bool:
        return _entity_key(entity) in self._graph


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _entity_key(entity: str) -> str:
    """Canonicalise an entity string to a graph node key."""
    return entity.strip().lower()


def _infer_document_id(chunk_id: str) -> str:
    """Best-effort extraction of a document id from a chunk id.

    The convention ``doc-xxx::chunk-yyy`` is used when chunks are
    registered via the ingestion pipeline.  Falls back to the empty
    string.
    """
    if not chunk_id:
        return ""
    # Heuristic: chunk ids often embed the document id
    if "::" in chunk_id:
        return chunk_id.split("::", 1)[0]
    return ""


def _chunk_allowed(chunk_id: str, allow_docs: set[str] | None) -> bool:
    """Return True when *chunk_id* passes the document filter."""
    if not chunk_id:
        return False  # empty chunk_ids are never valid
    if allow_docs is None:
        return True
    doc_id = _infer_document_id(chunk_id)
    if doc_id:
        return doc_id in allow_docs
    # If we can't determine the doc id, allow it (conservative)
    return True


def _json_default(obj: object) -> object:
    """JSON default for set serialisation."""
    if isinstance(obj, set):
        return sorted(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")
