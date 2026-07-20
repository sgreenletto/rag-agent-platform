"""Validated datasets for offline retrieval evaluation."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class RetrievalEvaluationCase:
    """One question with retrieval relevance judgments."""

    case_id: str
    question: str
    relevant_chunk_ids: tuple[str, ...]
    relevant_document_ids: tuple[str, ...] = ()
    document_ids: tuple[str, ...] = ()
    answerable: bool = True
    category: str = "uncategorized"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.case_id.strip():
            raise ValueError("case_id must not be empty")
        if not self.question.strip():
            raise ValueError("question must not be empty")
        if self.answerable and not self.relevant_chunk_ids:
            raise ValueError("answerable cases must define relevant_chunk_ids")
        if len(set(self.relevant_chunk_ids)) != len(self.relevant_chunk_ids):
            raise ValueError("relevant_chunk_ids must not contain duplicates")
        if any(not value.strip() for value in self.relevant_chunk_ids):
            raise ValueError("relevant_chunk_ids must not contain empty values")
        if not self.answerable and self.relevant_chunk_ids:
            raise ValueError("unanswerable cases must not define relevant_chunk_ids")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "RetrievalEvaluationCase":
        """Build a case from one decoded JSON object."""
        try:
            answerable = value.get("answerable", True)
            if not isinstance(answerable, bool):
                raise TypeError("answerable must be a boolean")
            return cls(
                case_id=_require_string(value["case_id"], "case_id"),
                question=_require_string(value["question"], "question"),
                relevant_chunk_ids=_require_string_list(
                    value["relevant_chunk_ids"], "relevant_chunk_ids"
                ),
                relevant_document_ids=_require_string_list(
                    value.get("relevant_document_ids", []), "relevant_document_ids"
                ),
                document_ids=_require_string_list(value.get("document_ids", []), "document_ids"),
                answerable=answerable,
                category=_require_string(value.get("category", "uncategorized"), "category"),
                metadata=_require_dict(value.get("metadata", {}), "metadata"),
            )
        except (KeyError, TypeError) as error:
            raise ValueError(f"invalid retrieval evaluation case: {error}") from error


def load_retrieval_dataset(path: str | Path) -> list[RetrievalEvaluationCase]:
    """Load a JSON array or JSONL file and reject duplicate case IDs."""
    dataset_path = Path(path)
    if not dataset_path.exists() or not dataset_path.is_file():
        raise FileNotFoundError(f"evaluation dataset does not exist: {dataset_path}")
    text = dataset_path.read_text(encoding="utf-8-sig")
    if not text.strip():
        raise ValueError("evaluation dataset must not be empty")

    if dataset_path.suffix.lower() == ".jsonl":
        records: list[Any] = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSONL at line {line_number}: {error.msg}") from error
    elif dataset_path.suffix.lower() == ".json":
        try:
            records = json.loads(text)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSON dataset: {error.msg}") from error
        if not isinstance(records, list):
            raise ValueError("JSON evaluation dataset must contain an array")
    else:
        raise ValueError("evaluation dataset must use .json or .jsonl")

    cases = [
        RetrievalEvaluationCase.from_dict(record)
        if isinstance(record, dict)
        else _raise_non_object(index)
        for index, record in enumerate(records, start=1)
    ]
    case_ids = [case.case_id for case in cases]
    if len(set(case_ids)) != len(case_ids):
        raise ValueError("evaluation dataset contains duplicate case_id values")
    if not cases:
        raise ValueError("evaluation dataset must contain at least one case")
    return cases


def _raise_non_object(index: int) -> RetrievalEvaluationCase:
    raise ValueError(f"evaluation case {index} must be a JSON object")


def _require_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    return value


def _require_string_list(value: Any, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise TypeError(f"{field_name} must be a list of strings")
    return tuple(value)


def _require_dict(value: Any, field_name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{field_name} must be an object")
    return dict(value)
