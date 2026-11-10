"""Pluggable embedder backends: ONNX pooling, the provider selector, and the API request/response path.

No network and no ONNX model needed — the pooling is tested on the numpy helper directly, and each API
backend is driven through a real ``httpx`` client backed by ``httpx.MockTransport`` (so the full
``_http()`` / ``_post()`` path and the API key reaching the request headers are exercised, with a mock
handler standing in for the network). ``httpx`` is optional (the ``embeddings-api`` extra), so the
mock-transport tests ``importorskip`` it; the pooling / selector / fingerprint tests don't need it.
"""

import json

import numpy as np
import pytest
from bausatz import Component

from tarnrag.core.engine.config import EmbeddingSettings
from tarnrag.core.resources.embedder import Embedder, HashEmbedder, OnnxEmbedder
from tarnrag.core.resources.embedder_api import (
    GeminiEmbedder,
    OpenAIEmbedder,
    VoyageEmbedder,
)
from tarnrag.core.resources.resource import Resource


def test_hash_embedder_is_deterministic_offline():
    """The model-free `hash` backend: deterministic, right dim, L2-normalized, no model/network."""
    emb = Embedder.create(EmbeddingSettings(provider="hash"), 384)
    assert isinstance(emb, HashEmbedder) and emb.dim() == 384 and emb.identity() == "hash:384"

    v = emb.embed_query("the eiffel tower is in paris")
    assert len(v) == 384
    assert abs(np.linalg.norm(v) - 1.0) < 1e-5  # L2-normalized
    assert emb.embed_query("the eiffel tower is in paris") == v  # deterministic
    assert emb.embed_passages(["a", "b"]) == [emb.embed_query("a"), emb.embed_query("b")]


def test_hash_embedder_identity_and_meta():
    emb = HashEmbedder(embedding_dim=8)
    meta = emb.embed_meta()
    assert meta["embedding_provider"] == "hash" and meta["embedding_dim"] == "8"
    assert meta["embedding_config_fingerprint"] == emb.config_fingerprint()
    assert HashEmbedder(embedding_dim=8).config_fingerprint() == emb.config_fingerprint()  # stable
    assert HashEmbedder(embedding_dim=16).config_fingerprint() != emb.config_fingerprint()  # dim-sensitive


# ---------------- ONNX pooling / normalize (now config-driven) ----------------

def _onnx(pooling="mean", normalize="l2"):
    return OnnxEmbedder("d", model_id="m", pooling=pooling, normalize=normalize)


def test_pooling_modes():
    # out [B=1, L=3, dim=2]; the 3rd token is padding (mask 0) — Qwen-style 'last' must ignore it.
    out = np.array([[[1.0, 0.0], [3.0, 0.0], [9.0, 9.0]]], dtype=np.float32)
    mask = np.array([[1, 1, 0]], dtype=np.int64)
    assert _onnx("mean")._pool(out, mask).tolist() == [[2.0, 0.0]]  # ([1,0]+[3,0])/2
    assert _onnx("cls")._pool(out, mask).tolist() == [[1.0, 0.0]]   # first token
    assert _onnx("last")._pool(out, mask).tolist() == [[3.0, 0.0]]  # last non-pad token, not the [9,9] pad


def test_pooling_rejects_unknown():
    with pytest.raises(ValueError, match="unknown pooling"):
        _onnx("bogus")._pool(np.zeros((1, 1, 2), np.float32), np.ones((1, 1), np.int64))


def test_normalize_modes():
    pooled = np.array([[3.0, 4.0]], dtype=np.float32)  # norm 5
    assert _onnx(normalize="l2")._normalize(pooled).tolist()[0] == pytest.approx([0.6, 0.8])  # unit length
    assert _onnx(normalize="none")._normalize(pooled).tolist() == [[3.0, 4.0]]


# ---------------- the provider selector ----------------

def test_embedder_create_dispatch():
    assert isinstance(Embedder.create(EmbeddingSettings(), 384), OnnxEmbedder)  # local default
    cases = [("openai", OpenAIEmbedder, 1536), ("voyage", VoyageEmbedder, 1024), ("gemini", GeminiEmbedder, 3072)]
    for provider, cls, dim in cases:
        e = Embedder.create(EmbeddingSettings(provider=provider, model="m"), dim)
        assert isinstance(e, cls) and e.dim() == dim


def test_api_embedders_are_resources_not_components():
    e = OpenAIEmbedder(model="m", embedding_dim=2, api_key="k")
    assert isinstance(e, (Resource, Embedder)) and not isinstance(e, Component)


def test_provider_is_part_of_the_fingerprint():
    a = Embedder.create(EmbeddingSettings(provider="openai", model="x"), 1536)
    b = Embedder.create(EmbeddingSettings(provider="voyage", model="x"), 1536)
    assert a.config_fingerprint() != b.config_fingerprint()  # same model, different backend -> different index
    assert a.embed_meta()["embedding_provider"] == "openai"
    assert a.embed_meta()["embedding_config_fingerprint"] == a.config_fingerprint()


# ---------------- API request/response path (real httpx client over a mock transport) ----------------

def _mock_client(handler):
    """A real httpx client whose transport runs ``handler(request) -> httpx.Response`` — no network, but
    the embedder's ``_http()`` / ``_post()`` (and the request it builds) run for real."""
    import httpx

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_openai_full_http_path():
    httpx = pytest.importorskip("httpx")  # optional dep (the embeddings-api extra)
    cap = {}

    def handler(request):
        cap.update(url=str(request.url), method=request.method, headers=request.headers,
                   body=json.loads(request.content))
        # out-of-order indices (exercise the sort) + known vectors (check parsing/normalization)
        return httpx.Response(200, json={"data": [
            {"index": 1, "embedding": [0.0, 2.0]}, {"index": 0, "embedding": [3.0, 0.0]},
        ]})

    e = OpenAIEmbedder(model="text-embedding-3-small", embedding_dim=2, api_key="sk-test")
    e._client = _mock_client(handler)
    vecs = e.embed_passages(["a", "b"])
    assert cap["method"] == "POST" and cap["url"].endswith("/embeddings")
    assert cap["headers"]["authorization"] == "Bearer sk-test"  # the key reached the real request
    assert cap["body"] == {"model": "text-embedding-3-small", "input": ["a", "b"], "dimensions": 2}
    assert vecs == [[1.0, 0.0], [0.0, 1.0]]  # index-sorted, then L2-normalized


def test_voyage_full_http_path():
    httpx = pytest.importorskip("httpx")
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [3.0, 4.0]}]})

    e = VoyageEmbedder(model="voyage-3.5", embedding_dim=2, api_key="vk")
    e._client = _mock_client(handler)
    qv = e.embed_query("q")
    e.embed_passages(["p"])
    assert bodies[0]["input_type"] == "query" and bodies[1]["input_type"] == "document"
    assert bodies[0]["output_dimension"] == 2
    assert qv == pytest.approx([0.6, 0.8])  # [3,4] L2-normalized


def test_gemini_full_http_path():
    httpx = pytest.importorskip("httpx")
    cap = {}

    def handler(request):
        cap.update(url=str(request.url), headers=request.headers, body=json.loads(request.content))
        return httpx.Response(200, json={"embeddings": [{"values": [0.0, 5.0]}]})

    e = GeminiEmbedder(model="gemini-embedding-001", embedding_dim=2, api_key="gk")
    e._client = _mock_client(handler)
    vec = e.embed_query("q")
    assert cap["url"].endswith("models/gemini-embedding-001:batchEmbedContents")
    req = cap["body"]["requests"][0]
    assert req["model"] == "models/gemini-embedding-001"
    assert req["taskType"] == "RETRIEVAL_QUERY" and req["outputDimensionality"] == 2
    assert cap["headers"]["x-goog-api-key"] == "gk"
    assert vec == [0.0, 1.0]  # L2-normalized


def test_api_key_falls_back_to_env_var(monkeypatch):
    httpx = pytest.importorskip("httpx")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")
    cap = {}

    def handler(request):
        cap["headers"] = request.headers
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [1.0, 0.0]}]})

    e = OpenAIEmbedder(model="m", embedding_dim=2)  # no explicit key -> env-var fallback
    e._client = _mock_client(handler)
    e.embed_query("q")
    assert cap["headers"]["authorization"] == "Bearer sk-from-env"


def test_api_key_is_required(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="API key"):
        OpenAIEmbedder(model="m", embedding_dim=2).embed_passages(["x"])


def test_onnx_fingerprint_is_sensitive_to_contextualize_tables(tmp_path):
    """contextualize_tables is an embed-time index variant: it must change the fingerprint, so a
    contextualized index never opens under a non-contextualizing embedder (mirrors inject_header_path)."""
    (tmp_path / "tokenizer.json").write_text("{}")  # identity hashes the tokenizer file only
    base = OnnxEmbedder(str(tmp_path), model_id="m")
    ctx = OnnxEmbedder(str(tmp_path), model_id="m", contextualize_tables=True)
    assert base.config_fingerprint() != ctx.config_fingerprint()
    assert ctx.embed_meta()["contextualize_tables"] == "True"


# ---------------- MED-03 / MED-04 / HIGH-07: identity, prefixes, the tokenizer read, close ----------------

def _fingerprint_of(identity: dict) -> str:
    return Embedder._fingerprint(identity)


def test_api_and_hash_fingerprints_track_the_embed_time_flags_and_keep_their_defaults():
    """inject_header_path / contextualize_tables change the embedded text, so they must change the
    fingerprint — but only when ON: an index built with the defaults keeps its fingerprint (and opens)."""
    for provider in ("openai", "voyage", "gemini", "hash"):
        base = Embedder.create(EmbeddingSettings(provider=provider, model="m"), 8)
        default_identity = (
            {"provider": provider, "embedding_dim": 8, "normalize": "l2"}
            if provider == "hash"
            else {"provider": provider, "model": "m", "embedding_dim": 8, "normalize": "l2"}
        )
        assert base.config_fingerprint() == _fingerprint_of(default_identity), provider  # unchanged
        for flag in ("inject_header_path", "contextualize_tables"):
            on = Embedder.create(EmbeddingSettings(provider=provider, model="m", **{flag: True}), 8)
            assert on.config_fingerprint() != base.config_fingerprint(), (provider, flag)
            assert on.embed_meta()[flag] == "True"


def test_api_prefixes_reach_the_request_and_the_fingerprint():
    httpx = pytest.importorskip("httpx")
    bodies = []

    def handler(request):
        body = json.loads(request.content)
        bodies.append(body["input"])
        return httpx.Response(200, json={"data": [{"index": i, "embedding": [1.0, 0.0]} for i in range(len(body["input"]))]})

    settings = EmbeddingSettings(provider="openai", model="e5", api_key="k",
                                 query_prefix="query: ", passage_prefix="passage: ")
    e = Embedder.create(settings, 2)
    e._client = _mock_client(handler)
    e.embed_passages(["a", "b"])
    e.embed_query("q")
    assert bodies == [["passage: a", "passage: b"], ["query: q"]]
    plain = Embedder.create(EmbeddingSettings(provider="openai", model="e5"), 2)
    assert e.config_fingerprint() != plain.config_fingerprint()  # prefixes change every vector


def test_api_embedder_close_releases_its_client():
    pytest.importorskip("httpx")
    e = OpenAIEmbedder(model="m", embedding_dim=2, api_key="k")
    client = e._http()
    e.close()
    assert client.is_closed and e._client is None
    e.close()  # idempotent
    assert not e._http().is_closed  # a later embed reopens one
    e.close()


def test_onnx_tokenizer_is_hashed_once_and_a_missing_model_says_how_to_fix_it(tmp_path, monkeypatch):
    import tarnrag.core.resources.embedder as embedder_module

    (tmp_path / "tokenizer.json").write_text("{}")
    reads = []
    real = embedder_module.sha256_file
    monkeypatch.setattr(embedder_module, "sha256_file", lambda p: reads.append(p) or real(p))
    e = OnnxEmbedder(str(tmp_path), model_id="m")
    e.config_fingerprint()
    e.embed_meta()
    e.config_fingerprint()
    assert len(reads) == 1
    with pytest.raises(FileNotFoundError, match="scripts/fetch_model.py"):
        OnnxEmbedder(str(tmp_path / "absent"), model_id="m").config_fingerprint()
