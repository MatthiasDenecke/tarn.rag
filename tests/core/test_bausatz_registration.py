"""tarn.rag builds on the standalone ``bausatz`` component framework (PyPI) — the framework's own
tests live in its repo; here we pin only tarn.rag's side of the contract: importing the domain
packages registers every built-in component into bausatz's global factory, and (bausatz >= 0.3) the
containers declare their children as typed Configs, so a spec tree validates — and round-trips — whole."""

import pytest
from bausatz import ComponentFactory
from pydantic import ValidationError

import tarnrag.conversation
import tarnrag.generation
import tarnrag.ingestion
import tarnrag.retrieval
from tarnrag.core.engine.config import Settings
from tarnrag.retrieval.components.retriever import Retriever

# Importing the domain packages is what registers their components (import = register).
_DOMAIN_PACKAGES = (tarnrag.conversation, tarnrag.generation, tarnrag.ingestion, tarnrag.retrieval)


def test_domain_components_register_into_the_bausatz_factory():
    factory = ComponentFactory.get()
    for tag in (
        "pipeline", "LoadAndParse", "Chunk", "Embed", "structure_aware", "table_json",
        "retrieval_pipeline", "dense", "sparse", "rrf", "cross_encoder", "intent",
        "generation_pipeline", "single_hop", "decomposition", "answerability",
        "conversation_memory", "condense",
    ):
        assert tag in factory.registry, tag


def test_default_specs_validate_eagerly_as_whole_trees():
    """Every default composition in ``Settings`` validates in one pass, its children resolved to their
    concrete Configs (not left as raw dicts)."""
    settings = Settings()
    factory = ComponentFactory.get()
    for key, spec in settings.components.items():
        assert factory.validate(spec).class_name == spec["class_name"], key
    retrieval = factory.validate(settings.components["retrieval_pipeline"])
    assert all(isinstance(r, Retriever.Config) for r in retrieval.retrievers)


def test_a_foreign_child_is_rejected_at_validation():
    """A child from another family (a fuser in the retrievers slot) fails at validation, not build."""
    spec = {"class_name": "retrieval_pipeline", "retrievers": [{"class_name": "rrf"}]}
    with pytest.raises(ValidationError, match=r"retrievers\.0"):
        ComponentFactory.get().validate(spec)


def test_nested_components_round_trip_through_to_dict():
    """``to_dict`` serializes the whole tree — nested children included — back to a buildable spec."""
    factory = ComponentFactory.get()
    spec = {
        "class_name": "routing_retrieval_pipeline",
        "classifier": {"class_name": "structural"},
        "routes": {"lexical": {"class_name": "retrieval_pipeline", "retrievers": [{"class_name": "sparse"}]}},
    }
    router = factory.create(spec)
    rebuilt = factory.create(router.to_dict())
    assert rebuilt.to_dict() == router.to_dict()
    assert router.to_dict()["routes"]["lexical"]["retrievers"][0]["class_name"] == "sparse"
    checker = {"class_name": "cascading_grounding", "checkers": [{"class_name": "heuristic_grounding"}]}
    gen = factory.create({"class_name": "generation_pipeline", "grounding_checker": checker})
    assert gen.to_dict()["grounding_checker"]["checkers"][0]["class_name"] == "heuristic_grounding"
