"""The ``Resource`` base — the embedder/cross-encoder are injected models, siblings of ``Component``."""

from bausatz import Component

from tarnrag.core.engine.config import EmbeddingSettings, RerankSettings
from tarnrag.core.resources.cross_encoder import CrossEncoder, OnnxCrossEncoder
from tarnrag.core.resources.embedder import Embedder, OnnxEmbedder
from tarnrag.core.resources.resource import Resource


def test_models_are_resources_not_components():
    # Construction is cheap (the ONNX model loads lazily), so this needs no model on disk.
    emb = OnnxEmbedder.create(EmbeddingSettings(), 384)
    ce = OnnxCrossEncoder.create(RerankSettings())
    assert isinstance(emb, (Resource, Embedder)) and isinstance(ce, (Resource, CrossEncoder))
    # The whole point: a Resource is a sibling of Component, not a subclass of it.
    assert not isinstance(emb, Component) and not isinstance(ce, Component)


def test_identity_is_model_id_with_optional_revision():
    assert OnnxEmbedder("d", model_id="e").identity() == "e"
    assert OnnxCrossEncoder("d", model_id="m").identity() == "m"
    assert OnnxCrossEncoder("d", model_id="m", revision="r1").identity() == "m@r1"
