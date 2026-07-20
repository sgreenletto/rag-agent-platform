import json
from pathlib import Path

import pytest

from rag_agent_platform.evaluation.retrieval_dataset import (
    RetrievalEvaluationCase,
    load_retrieval_dataset,
)


def test_loads_jsonl_fixture() -> None:
    cases = load_retrieval_dataset(Path("tests/fixtures/retrieval_questions.jsonl"))

    assert [case.case_id for case in cases] == ["leave", "comparison", "unknown"]
    assert cases[0].document_ids == ("hr",)
    assert cases[1].relevant_chunk_ids == ("travel-policy", "hospitality-policy")
    assert cases[2].answerable is False


def test_loads_json_array(tmp_path: Path) -> None:
    path = tmp_path / "questions.json"
    path.write_text(
        json.dumps(
            [
                {
                    "case_id": "one",
                    "question": "question",
                    "relevant_chunk_ids": ["chunk"],
                }
            ]
        ),
        encoding="utf-8",
    )

    assert load_retrieval_dataset(path)[0].case_id == "one"


@pytest.mark.parametrize(("field", "value"), [("case_id", " "), ("question", " ")])
def test_case_rejects_empty_identity_fields(field: str, value: str) -> None:
    values = {"case_id": "id", "question": "question", "relevant_chunk_ids": ("chunk",)}
    values[field] = value

    with pytest.raises(ValueError, match=field):
        RetrievalEvaluationCase(**values)  # type: ignore[arg-type]


def test_answerable_case_requires_relevance_judgment() -> None:
    with pytest.raises(ValueError, match="relevant_chunk_ids"):
        RetrievalEvaluationCase("id", "question", ())


def test_dataset_rejects_duplicate_case_ids(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.json"
    record = {"case_id": "same", "question": "q", "relevant_chunk_ids": ["chunk"]}
    path.write_text(json.dumps([record, record]), encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate case_id"):
        load_retrieval_dataset(path)


def test_dataset_reports_invalid_jsonl_line(tmp_path: Path) -> None:
    path = tmp_path / "broken.jsonl"
    path.write_text('{"valid": true}\nnot-json', encoding="utf-8")

    with pytest.raises(ValueError, match="line 2"):
        load_retrieval_dataset(path)


def test_dataset_rejects_unsupported_extension(tmp_path: Path) -> None:
    path = tmp_path / "questions.txt"
    path.write_text("[]", encoding="utf-8")

    with pytest.raises(ValueError, match=".json or .jsonl"):
        load_retrieval_dataset(path)


def test_dataset_supports_utf8_bom(tmp_path: Path) -> None:
    path = tmp_path / "questions.json"
    path.write_text(
        '[{"case_id":"one","question":"q","relevant_chunk_ids":["chunk"]}]',
        encoding="utf-8-sig",
    )

    assert load_retrieval_dataset(path)[0].case_id == "one"


@pytest.mark.parametrize(
    ("field", "value"),
    [("answerable", "false"), ("relevant_chunk_ids", "chunk"), ("metadata", [])],
)
def test_dataset_rejects_wrong_json_field_types(tmp_path: Path, field: str, value: object) -> None:
    path = tmp_path / "questions.json"
    record: dict[str, object] = {
        "case_id": "one",
        "question": "q",
        "relevant_chunk_ids": ["chunk"],
    }
    record[field] = value
    path.write_text(json.dumps([record]), encoding="utf-8")

    with pytest.raises(ValueError):
        load_retrieval_dataset(path)


def test_unanswerable_case_rejects_relevant_chunks() -> None:
    with pytest.raises(ValueError, match="unanswerable"):
        RetrievalEvaluationCase("id", "question", ("chunk",), answerable=False)
