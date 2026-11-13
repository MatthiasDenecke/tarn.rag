"""Settings: grouped sub-models, env loading, and distributed-config validation."""

import pytest
from pydantic import ValidationError

from tarnrag.core.engine.config import Settings


def test_defaults_and_groups():
    s = Settings(_env_file=None)  # ignore the local .env, exercise defaults
    assert s.MODE == "embedded"
    assert s.EMBEDDING_DIMENSION == 384
    assert s.embedding.provider == "onnx"  # local ONNX by default
    assert s.embedding.model == "thenlper/gte-small"
    assert s.database.document_url.startswith("sqlite")  # zero-config embedded default


def test_llm_key_env_var_defaults_to_the_provider_standard():
    from tarnrag.core.engine.config import LLMSettings

    assert LLMSettings(provider="anthropic").key_env_var() == "ANTHROPIC_API_KEY"
    assert LLMSettings(provider="openai").key_env_var() == "OPENAI_API_KEY"
    assert LLMSettings(provider="openai", api_key_env="OPENAI_LLM_KEY").key_env_var() == "OPENAI_LLM_KEY"


def test_llm_resolved_api_key_prefers_explicit_then_the_named_env_var(monkeypatch):
    from tarnrag.core.engine.config import LLMSettings

    monkeypatch.delenv("OPENAI_LLM_KEY", raising=False)
    llm = LLMSettings(provider="openai", api_key_env="OPENAI_LLM_KEY")
    assert llm.resolved_api_key() == ""  # nothing set anywhere

    monkeypatch.setenv("OPENAI_LLM_KEY", "sk-from-custom-var")
    assert llm.resolved_api_key() == "sk-from-custom-var"  # read from the configured env var

    explicit = LLMSettings(provider="openai", api_key="sk-explicit", api_key_env="OPENAI_LLM_KEY")
    assert explicit.resolved_api_key() == "sk-explicit"  # an explicit key wins over the env var


def test_nested_env_loading(monkeypatch):
    monkeypatch.setenv("EMBEDDING__MODEL", "my/model")
    monkeypatch.setenv("EMBEDDING__BATCH_SIZE", "8")
    monkeypatch.setenv("EMBEDDING_DIMENSION", "768")
    s = Settings(_env_file=None)
    assert s.embedding.model == "my/model"
    assert s.embedding.batch_size == 8
    assert s.EMBEDDING_DIMENSION == 768  # cross-cutting field stays top-level/flat


def test_unknown_env_vars_are_ignored_not_fatal(monkeypatch):
    """A stale/unknown env var (e.g. a removed config group, or a project's own custom key name) must not
    crash Settings — extra='ignore'."""
    monkeypatch.setenv("CHUNKING__SIZE", "512")  # a group that no longer exists on Settings
    monkeypatch.setenv("OPENAI_LLM_KEY", "sk-custom")  # a non-standard, project-specific var
    s = Settings(_env_file=None)  # must not raise
    assert not hasattr(s, "chunking")
    assert s.embedding.model  # the real fields still load


def test_default_generation_reasoner_is_decomposition():
    """Phase 1: the shipped default generation pipeline uses the decomposition reasoner (best on average)."""
    from tarnrag.core.engine.config import GENERATION_PIPELINE

    spec = Settings(_env_file=None).components[GENERATION_PIPELINE]
    assert spec["reasoner"]["class_name"] == "decomposition"


def test_default_retrieval_pipeline_is_hybrid():
    """P2: the shipped default retrieval pipeline is HYBRID — dense + sparse, RRF-fused (measured: never
    lost on any segment, biggest win on table/layout corpora, free at query time)."""
    from tarnrag.core.engine.config import RETRIEVAL_PIPELINE

    spec = Settings(_env_file=None).components[RETRIEVAL_PIPELINE]
    assert [r["class_name"] for r in spec["retrievers"]] == ["dense", "sparse"]
    assert spec["fuser"]["class_name"] == "rrf"


def test_kwargs_override_nested():
    s = Settings(_env_file=None, embedding={"model": "k"}, EMBEDDING_DIMENSION=128)
    assert s.embedding.model == "k"
    assert s.EMBEDDING_DIMENSION == 128


def test_distributed_requires_postgres_and_queue():
    # distributed + the SQLite document default -> rejected
    with pytest.raises(ValidationError, match="Postgres DATABASE__DOCUMENT_URL"):
        Settings(_env_file=None, MODE="distributed", database={"queue_url": "postgresql://q"})
    # distributed + Postgres docs but no queue -> rejected
    with pytest.raises(ValidationError, match="DATABASE__QUEUE_URL"):
        Settings(_env_file=None, MODE="distributed", database={"document_url": "postgresql://docs"})
    # fully configured distributed -> ok
    s = Settings(
        _env_file=None, MODE="distributed",
        database={"document_url": "postgresql://docs", "queue_url": "postgresql://q"},
    )
    assert s.MODE == "distributed"
    # embedded with the SQLite default -> ok (no validation trip)
    assert Settings(_env_file=None).MODE == "embedded"


def test_embedded_rejects_postgres_document_url():
    # embedded + Postgres document store -> rejected (embedded is SQLite / single-process)
    with pytest.raises(ValidationError, match="must not use a Postgres"):
        Settings(_env_file=None, MODE="embedded", database={"document_url": "postgresql://docs"})
    # embedded + the SQLite default -> ok
    assert Settings(_env_file=None, MODE="embedded").database.document_url.startswith("sqlite")


def test_conversation_memory_is_default_filled_and_off():
    """The fifth components key is always present (a Settings is self-complete) and DISABLED — memory
    is opt-in, so an unconfigured deployment stays stateless."""
    from tarnrag.core.engine.config import CONVERSATION_MEMORY

    spec = Settings(_env_file=None).components[CONVERSATION_MEMORY]
    assert spec == {"class_name": "conversation_memory", "enabled": False}


def test_a_supplied_conversation_memory_spec_wins_over_the_default():
    from tarnrag.core.engine.config import CONVERSATION_MEMORY

    supplied = {"class_name": "conversation_memory", "enabled": True, "max_turns": 3}
    settings = Settings(_env_file=None, components={CONVERSATION_MEMORY: supplied})
    assert settings.components[CONVERSATION_MEMORY] == supplied
    assert "retrieval_pipeline" in settings.components  # the other defaults are still filled alongside


def test_conversation_memory_is_switched_from_a_yaml_file(tmp_path):
    """The on/off switch is plain config: ``enabled`` in the YAML, read by ``Settings.from_file``."""
    from tarnrag.core.engine.config import CONVERSATION_MEMORY

    def load(enabled: str) -> dict:
        cfg = tmp_path / f"memory-{enabled}.yaml"
        cfg.write_text(
            "components:\n"
            "  conversation_memory:\n"
            "    class_name: conversation_memory\n"
            f"    enabled: {enabled}\n"
            "    max_turns: 4\n"
        )
        return Settings.from_file(cfg).components[CONVERSATION_MEMORY]

    assert load("true")["enabled"] is True and load("true")["max_turns"] == 4
    assert load("false")["enabled"] is False
