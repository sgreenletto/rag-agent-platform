"""Run offline retrieval evaluation against a project-provided Retriever factory."""

import argparse
import importlib
import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import cast

from rag_agent_platform.evaluation import RetrievalEvaluationRunner, load_retrieval_dataset
from rag_agent_platform.retrieval import BaseRetriever

RetrieverFactory = Callable[[], Mapping[str, BaseRetriever]]


def load_factory(reference: str) -> RetrieverFactory:
    """Load a zero-argument `module:function` Retriever factory."""
    try:
        module_name, function_name = reference.split(":", maxsplit=1)
    except ValueError as error:
        raise ValueError("factory must use module:function format") from error
    factory = getattr(importlib.import_module(module_name), function_name, None)
    if not callable(factory):
        raise ValueError(f"retriever factory is not callable: {reference}")
    return cast(RetrieverFactory, factory)


def validate_retrievers(value: object) -> Mapping[str, BaseRetriever]:
    """Validate the factory output before starting an evaluation run."""
    if not isinstance(value, Mapping) or not value:
        raise ValueError("retriever factory must return a non-empty mapping")
    for mode, retriever in value.items():
        if not isinstance(mode, str) or not mode.strip():
            raise ValueError("retriever mode names must be non-empty strings")
        if not isinstance(retriever, BaseRetriever):
            raise ValueError(f"retriever mode {mode!r} does not implement BaseRetriever")
    return cast(Mapping[str, BaseRetriever], value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare retrieval modes offline")
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument(
        "--factory", required=True, help="module:function returning named Retrievers"
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    cases = load_retrieval_dataset(args.dataset)
    reports = RetrievalEvaluationRunner().compare(
        validate_retrievers(load_factory(args.factory)()), cases, top_k=args.top_k
    )
    payload = {mode: report.to_dict() for mode, report in reports.items()}
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)


if __name__ == "__main__":
    main()
