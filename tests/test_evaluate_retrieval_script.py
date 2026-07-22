import pytest

from rag_agent_platform.retrieval.mock import MockRetriever
from scripts.evaluate_retrieval import load_factory, validate_retrievers


def test_script_rejects_factory_without_function_separator() -> None:
    with pytest.raises(ValueError, match="module:function"):
        load_factory("module_only")


def test_script_rejects_non_callable_factory() -> None:
    with pytest.raises(ValueError, match="not callable"):
        load_factory("rag_agent_platform:__version__")


def test_script_validates_named_retriever_mapping() -> None:
    retrievers = {"naive": MockRetriever("naive")}

    assert validate_retrievers(retrievers) == retrievers


@pytest.mark.parametrize("value", [{}, [], {"naive": object()}, {"": MockRetriever("naive")}])
def test_script_rejects_invalid_factory_output(value: object) -> None:
    with pytest.raises(ValueError, match="retriever|mode"):
        validate_retrievers(value)
