"""HTTP embedding-API backends — OpenAI(-compatible), Voyage, and Gemini.

Each is an ``Embedder`` (a ``Resource``, like ``OnnxEmbedder``) selected via ``EmbeddingSettings.provider``
and built by ``Embedder.create``. They share ``_ApiEmbedder``: batched POST → parse → **L2-normalize** (so
dense KNN stays cosine-equivalent across backends), plus a provider/model/dim identity that rides the
index fingerprint. The query/passage distinction is handled the way each API expects — OpenAI doesn't
distinguish; Voyage uses ``input_type``; Gemini uses ``taskType``.

``httpx`` is imported lazily (the ``embeddings-api`` extra). The real API paths are **not** exercised by
the test suite (they need a key + network); the request-shaping / parsing / identity logic is unit-tested
by stubbing the one ``_post`` seam — the same honest gating as the ONNX model and the Postgres dialect.
"""

from __future__ import annotations

import os
from abc import abstractmethod
from typing import Any

from tarnrag.core.engine.config import EmbeddingSettings
from tarnrag.core.resources.embedder import Embedder


class _ApiEmbedder(Embedder):
    """Shared base for HTTP embedding APIs: batching, L2-normalization, identity/fingerprint, and the
    single ``_post`` seam. Subclasses set ``PROVIDER`` / ``DEFAULT_BASE_URL`` / ``API_KEY_ENV`` and
    implement ``_embed_batch`` (the provider's request shape + response parse).

    ``query_prefix`` / ``passage_prefix`` are prepended exactly as the ONNX backend does (an
    OpenAI-compatible endpoint serving an E5 / BGE model needs them; Voyage / Gemini express the same
    distinction with ``input_type`` / ``taskType`` and usually leave them empty). Prefixes and the
    embed-time ``inject_header_path`` / ``contextualize_tables`` flags ride the fingerprint when set."""

    PROVIDER: str = ""
    DEFAULT_BASE_URL: str = ""
    API_KEY_ENV: str = ""

    def __init__(
        self,
        *,
        model: str,
        embedding_dim: int,
        api_key: str = "",
        base_url: str = "",
        batch_size: int = 32,
        timeout: float = 30.0,
        query_prefix: str = "",
        passage_prefix: str = "",
        inject_header_path: bool = False,
        contextualize_tables: bool = False,
    ):
        self.model = model
        self.embedding_dim = embedding_dim
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix
        self.inject_header_path = inject_header_path
        self.contextualize_tables = contextualize_tables
        self._api_key = api_key
        self._base_url = (base_url or self.DEFAULT_BASE_URL).rstrip("/")
        self.batch_size = batch_size
        self.timeout = timeout
        self._client = None

    @classmethod
    def create(cls, embedding: EmbeddingSettings, embedding_dimension: int) -> _ApiEmbedder:
        return cls(
            model=embedding.model,
            embedding_dim=embedding_dimension,
            api_key=embedding.api_key,
            base_url=embedding.api_base_url,
            batch_size=embedding.batch_size,
            timeout=embedding.api_timeout,
            query_prefix=embedding.query_prefix,
            passage_prefix=embedding.passage_prefix,
            inject_header_path=embedding.inject_header_path,
            contextualize_tables=embedding.contextualize_tables,
        )

    # ---------------- identity (no network) ----------------

    def dim(self) -> int:
        return self.embedding_dim

    def identity(self) -> str:
        return f"{self.PROVIDER}:{self.model}"

    def _identity(self) -> dict[str, Any]:
        """Provider / model / dim / normalization, plus — only when set, so an index built with the
        defaults keeps its fingerprint — the prefixes and the embed-time text transforms (each changes the
        vectors)."""
        return {
            "provider": self.PROVIDER,
            "model": self.model,
            "embedding_dim": self.embedding_dim,
            "normalize": "l2",
            **({"query_prefix": self.query_prefix} if self.query_prefix else {}),
            **({"passage_prefix": self.passage_prefix} if self.passage_prefix else {}),
            **self._embed_time_identity(
                inject_header_path=self.inject_header_path, contextualize_tables=self.contextualize_tables
            ),
        }

    def config_fingerprint(self) -> str:
        return self._fingerprint(self._identity())

    def embed_meta(self) -> dict[str, str]:
        ident = self._identity()
        return {
            "embedding_provider": ident["provider"],
            "embedding_model_id": ident["model"],
            "embedding_dim": str(ident["embedding_dim"]),
            "normalize": ident["normalize"],
            "query_prefix": self.query_prefix,
            "passage_prefix": self.passage_prefix,
            "inject_header_path": str(self.inject_header_path),
            "contextualize_tables": str(self.contextualize_tables),
            "embedding_config_fingerprint": self.config_fingerprint(),
        }

    # ---------------- inference ----------------

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        return self._embed([self.passage_prefix + t for t in texts], is_query=False)

    def embed_query(self, text: str) -> list[float]:
        return self._embed([self.query_prefix + text], is_query=True)[0]

    def close(self) -> None:
        """Close the ``httpx`` client, if one was opened (idempotent; a later embed reopens one)."""
        if self._client is not None:
            client, self._client = self._client, None
            client.close()

    def _embed(self, texts: list[str], *, is_query: bool) -> list[list[float]]:
        if not texts:
            return []
        raw: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            raw.extend(self._embed_batch(texts[i : i + self.batch_size], is_query=is_query))
        return self._l2_normalize(raw)

    @abstractmethod
    def _embed_batch(self, texts: list[str], *, is_query: bool) -> list[list[float]]:
        """One API request for ``texts``; return raw vectors in input order (the base L2-normalizes)."""

    # ---------------- http (the one stubbable seam) ----------------

    def _key(self) -> str:
        key = self._api_key or os.environ.get(self.API_KEY_ENV, "")
        if not key:
            raise ValueError(
                f"{self.PROVIDER} embedder needs an API key — set EMBEDDING__API_KEY or {self.API_KEY_ENV}"
            )
        return key

    def _http(self):
        if self._client is None:
            try:
                import httpx
            except ImportError as e:  # pragma: no cover - exercised only without the extra
                raise RuntimeError(
                    "API embedders need httpx — install the 'embeddings-api' extra (pip install '.[embeddings-api]')"
                ) from e
            self._client = httpx.Client(timeout=self.timeout)
        return self._client

    def _post(self, url: str, body: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
        """POST JSON and return the parsed body. The single seam tests stub to avoid the network."""
        resp = self._http().post(url, json=body, headers=headers)
        resp.raise_for_status()
        return resp.json()

    def _bearer_headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._key()}", "Content-Type": "application/json"}

    @staticmethod
    def _l2_normalize(vectors: list[list[float]]) -> list[list[float]]:
        import numpy as np

        arr = np.asarray(vectors, dtype=np.float32)
        norms = np.clip(np.linalg.norm(arr, axis=1, keepdims=True), 1e-12, None)
        return (arr / norms).tolist()


class OpenAIEmbedder(_ApiEmbedder):
    """OpenAI (and OpenAI-compatible) ``/embeddings``. Requests ``dimensions`` so the output matches
    ``EMBEDDING_DIMENSION`` (text-embedding-3 / compatible models); query and passage are embedded
    identically (the API has no input-type distinction)."""

    PROVIDER = "openai"
    DEFAULT_BASE_URL = "https://api.openai.com/v1"
    API_KEY_ENV = "OPENAI_API_KEY"

    def _embed_batch(self, texts: list[str], *, is_query: bool) -> list[list[float]]:
        body = {"model": self.model, "input": texts, "dimensions": self.embedding_dim}
        data = self._post(f"{self._base_url}/embeddings", body, self._bearer_headers())
        return [row["embedding"] for row in sorted(data["data"], key=lambda r: r["index"])]


class VoyageEmbedder(_ApiEmbedder):
    """Voyage AI ``/embeddings`` (Anthropic's recommended embeddings partner). Uses ``input_type``
    (query/document) and ``output_dimension`` (voyage-3.5 / voyage-3-large)."""

    PROVIDER = "voyage"
    DEFAULT_BASE_URL = "https://api.voyageai.com/v1"
    API_KEY_ENV = "VOYAGE_API_KEY"

    def _embed_batch(self, texts: list[str], *, is_query: bool) -> list[list[float]]:
        body = {
            "model": self.model,
            "input": texts,
            "input_type": "query" if is_query else "document",
            "output_dimension": self.embedding_dim,
        }
        data = self._post(f"{self._base_url}/embeddings", body, self._bearer_headers())
        return [row["embedding"] for row in sorted(data["data"], key=lambda r: r["index"])]


class GeminiEmbedder(_ApiEmbedder):
    """Google Gemini ``:batchEmbedContents`` (the model MOTHRAG used). Uses ``taskType``
    (RETRIEVAL_QUERY / RETRIEVAL_DOCUMENT) and ``outputDimensionality``; the key rides the
    ``x-goog-api-key`` header."""

    PROVIDER = "gemini"
    DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
    API_KEY_ENV = "GEMINI_API_KEY"

    def _embed_batch(self, texts: list[str], *, is_query: bool) -> list[list[float]]:
        model_path = self.model if self.model.startswith("models/") else f"models/{self.model}"
        task = "RETRIEVAL_QUERY" if is_query else "RETRIEVAL_DOCUMENT"
        body = {
            "requests": [
                {
                    "model": model_path,
                    "content": {"parts": [{"text": t}]},
                    "taskType": task,
                    "outputDimensionality": self.embedding_dim,
                }
                for t in texts
            ]
        }
        url = f"{self._base_url}/{model_path}:batchEmbedContents"
        data = self._post(url, body, {"x-goog-api-key": self._key(), "Content-Type": "application/json"})
        return [e["values"] for e in data["embeddings"]]


EMBEDDER_PROVIDERS: dict[str, type[_ApiEmbedder]] = {
    "openai": OpenAIEmbedder,
    "voyage": VoyageEmbedder,
    "gemini": GeminiEmbedder,
}
