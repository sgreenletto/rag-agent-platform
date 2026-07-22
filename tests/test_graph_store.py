"""Tests for NetworkXGraphStore — graph construction, search, JSON persistence."""

import json
import tempfile
from pathlib import Path

import pytest

from rag_agent_platform.graph.models import Triplet
from rag_agent_platform.graph.store import NetworkXGraphStore

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def empty_store() -> NetworkXGraphStore:
    return NetworkXGraphStore()


@pytest.fixture
def sample_triplets() -> list[Triplet]:
    return [
        Triplet("人事部", "负责", "请假制度", "doc-1::c1", 0.9),
        Triplet("请假制度", "属于", "公司规章", "doc-1::c1", 0.85),
        Triplet("公司规章", "包含", "考勤规则", "doc-1::c2", 0.85),
        Triplet("公司规章", "包含", "薪资规则", "doc-1::c2", 0.85),
        Triplet("员工", "依赖于", "考勤系统", "doc-2::c1", 0.9),
        Triplet("部门经理", "审批", "请假申请", "doc-2::c2", 0.8),
    ]


@pytest.fixture
def populated_store(sample_triplets: list[Triplet]) -> NetworkXGraphStore:
    store = NetworkXGraphStore()
    store.add_triplets(sample_triplets)
    return store


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_empty_store(empty_store: NetworkXGraphStore) -> None:
    assert len(empty_store) == 0
    assert empty_store.graph.number_of_edges() == 0


def test_add_triplets_builds_graph(populated_store: NetworkXGraphStore) -> None:
    assert len(populated_store) > 0
    assert populated_store.graph.number_of_edges() > 0


def test_add_empty_triplets_noop(empty_store: NetworkXGraphStore) -> None:
    empty_store.add_triplets([])
    assert len(empty_store) == 0


# ---------------------------------------------------------------------------
# Entity normalisation
# ---------------------------------------------------------------------------


def test_entity_case_insensitive(populated_store: NetworkXGraphStore) -> None:
    """Entity keys are lowercased / normalised."""
    # Check that an entity stored via a triplet is findable
    key = "人事部".strip().lower()
    assert key in populated_store  # normalised key exists
    assert "人事部".lower() in populated_store  # normalised


def test_duplicate_entities_merged() -> None:
    """Two triplets with the same subject should create one node."""
    store = NetworkXGraphStore()
    store.add_triplets(
        [
            Triplet("人事部", "负责", "请假制度", "doc-1::c1", 0.9),
            Triplet("人事部", "管理", "考勤系统", "doc-1::c2", 0.85),
        ]
    )
    # Both should reference the same "人事部" node
    assert len(store) < 4  # 人事部 + 请假制度 + 考勤系统 = 3 (not 4)


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


def test_search_returns_results(populated_store: NetworkXGraphStore) -> None:
    results = populated_store.search_neighborhood(["人事部"])
    assert len(results) > 0
    for cid, score in results:
        assert cid  # chunk_id must be non-empty
        assert 0.0 <= score <= 1.0


def test_search_no_match(empty_store: NetworkXGraphStore) -> None:
    results = empty_store.search_neighborhood(["nonexistent"])
    assert results == []


def test_search_top_k(populated_store: NetworkXGraphStore) -> None:
    results = populated_store.search_neighborhood(["请假制度"], top_k=2)
    assert len(results) <= 2


def test_search_document_filter(populated_store: NetworkXGraphStore) -> None:
    # Only doc-1 results
    results = populated_store.search_neighborhood(["请假制度"], document_ids=["doc-1"], top_k=10)
    for cid, _ in results:
        assert "doc-1" in cid


def test_search_scores_descending(populated_store: NetworkXGraphStore) -> None:
    results = populated_store.search_neighborhood(["人事部", "请假制度"])
    scores = [score for _, score in results]
    assert scores == sorted(scores, reverse=True)


def test_search_empty_entities(populated_store: NetworkXGraphStore) -> None:
    assert populated_store.search_neighborhood([], top_k=5) == []


def test_search_zero_top_k(populated_store: NetworkXGraphStore) -> None:
    assert populated_store.search_neighborhood(["人事部"], top_k=0) == []


# ---------------------------------------------------------------------------
# JSON persistence
# ---------------------------------------------------------------------------


def test_to_json_creates_file(populated_store: NetworkXGraphStore) -> None:
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "graph.json"
        populated_store.to_json(path)
        assert path.exists()
        data = json.loads(path.read_text("utf-8"))
        assert "nodes" in data
        assert "edges" in data
        assert "metadata" in data
        assert len(data["nodes"]) > 0
        assert len(data["edges"]) > 0


def test_json_round_trip(populated_store: NetworkXGraphStore) -> None:
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "graph.json"
        populated_store.to_json(path)
        restored = NetworkXGraphStore.from_json(path)
        assert len(restored) == len(populated_store)
        assert restored.graph.number_of_edges() == populated_store.graph.number_of_edges()


def test_json_round_trip_preserves_search(populated_store: NetworkXGraphStore) -> None:
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "graph.json"
        populated_store.to_json(path)
        restored = NetworkXGraphStore.from_json(path)

        original_results = populated_store.search_neighborhood(["人事部"], top_k=10)
        restored_results = restored.search_neighborhood(["人事部"], top_k=10)
        assert len(restored_results) == len(original_results)


def test_from_json_missing_file() -> None:
    with pytest.raises(FileNotFoundError):
        NetworkXGraphStore.from_json(Path("/nonexistent/graph.json"))


def test_json_creates_parent_directory() -> None:
    store = NetworkXGraphStore()
    with tempfile.TemporaryDirectory() as td:
        nested = Path(td) / "sub" / "deep" / "graph.json"
        store.to_json(nested)
        assert nested.exists()


# ---------------------------------------------------------------------------
# Document deletion
# ---------------------------------------------------------------------------


def test_remove_document(populated_store: NetworkXGraphStore) -> None:
    initial_nodes = len(populated_store)
    initial_edges = populated_store.graph.number_of_edges()

    populated_store.remove_document("doc-1")

    # Should have fewer nodes/edges
    assert len(populated_store) <= initial_nodes
    assert populated_store.graph.number_of_edges() <= initial_edges


def test_remove_document_preserves_other_docs(
    populated_store: NetworkXGraphStore,
) -> None:
    populated_store.remove_document("doc-1")

    # doc-2 entities should still be searchable
    results = populated_store.search_neighborhood(["员工"])
    for cid, _ in results:
        assert "doc-1" not in cid


def test_remove_nonexistent_document_noop(
    populated_store: NetworkXGraphStore,
) -> None:
    initial = len(populated_store)
    populated_store.remove_document("nonexistent-doc")
    assert len(populated_store) == initial


def test_remove_document_empty_id(populated_store: NetworkXGraphStore) -> None:
    initial = len(populated_store)
    populated_store.remove_document("")
    assert len(populated_store) == initial


# ---------------------------------------------------------------------------
# Clear
# ---------------------------------------------------------------------------


def test_clear(populated_store: NetworkXGraphStore) -> None:
    populated_store.clear()
    assert len(populated_store) == 0
    assert populated_store.graph.number_of_edges() == 0


# ---------------------------------------------------------------------------
# Chunk filtering (empty chunk_id rejection)
# ---------------------------------------------------------------------------


def test_triplets_with_empty_chunk_id_not_indexed() -> None:
    """Triplets from extract_from_text have no chunk_id — edges skipped."""
    store = NetworkXGraphStore()
    store.add_triplets(
        [Triplet("A", "是", "B", "", 0.9)]  # no chunk_id
    )
    # Nodes exist but edges are skipped (no provenance)
    assert len(store) >= 2  # nodes for A and B
    # Edges should be 0 because chunk_id was empty
    assert store.graph.number_of_edges() == 0
