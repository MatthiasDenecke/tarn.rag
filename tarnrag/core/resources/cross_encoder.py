"""The cross-encoder reranking model — a query × passage relevance scorer.

Retrieval-only (ingestion never reranks), but it lives next to the ``Embedder`` because it's the same
kind of thing: a heavy ONNX model injected at query time via the ``RetrievalContext`` and faked in tests.
Unlike a bi-encoder embedder it scores a (query, passage) **pair** jointly (one forward pass per pair),
which is more accurate but too costly for first-pass retrieval — hence it only re-scores the already
retrieved/merged shortlist.

Heavy deps (``onnxruntime``, ``tokenizers``) load lazily on the first ``score`` so import stays cheap.

**Caveat:** ``OnnxCrossEncoder`` targets the standard sentence-transformers cross-encoder export (a single
regression logit per pair, e.g. ``cross-encoder/ms-marco-MiniLM-L-6-v2``). Like the Postgres sparse path,
it is the production code path but is *not* exercised by the test suite (which uses a fake scorer); it
needs a local model dir to run.
"""

from __future__ import annotations

import threading
from abc import abstractmethod
from pathlib import Path

from tarnrag.core.engine.config import RerankSettings
from tarnrag.core.resources.resource import Resource


class CrossEncoder(Resource):
    """Port for a cross-encoder relevance scorer — a ``Resource`` injected at query time like the
    ``Embedder`` (a model the reranker consumes, not a composed ``Component``)."""

    @abstractmethod
    def score(self, query: str, passages: list[str]) -> list[float]:
        """One relevance score per passage for ``query`` (higher = more relevant), input order kept."""


class OnnxCrossEncoder(CrossEncoder):
    """ONNX Runtime cross-encoder (tokenize the query×passage pair → ONNX forward → relevance logit).
    Heavy deps load lazily on the first ``score``; construction is cheap (no model load)."""

    def __init__(
        self,
        model_dir: str,
        *,
        model_id: str,
        revision: str = "",
        max_length: int = 512,
        model_file: str = "model.onnx",
        tokenizer_file: str = "tokenizer.json",
    ):
        self.model_dir = Path(model_dir)
        self.model_id = model_id
        self.revision = revision
        self.max_length = max_length
        self.model_path = self.model_dir / model_file
        self.tokenizer_path = self.model_dir / tokenizer_file
        self._session = None
        self._load_lock = threading.Lock()  # concurrent first scores (off-loop) must load the model once
        self._tokenizer = None
        self._input_names: set[str] | None = None

    @classmethod
    def create(cls, rerank: RerankSettings) -> OnnxCrossEncoder:
        """Build the cross-encoder from its config slice (model loads lazily on first ``score``)."""
        return cls(
            rerank.model_dir,
            model_id=rerank.model,
            revision=rerank.revision,
            max_length=rerank.max_seq_length,
        )

    def identity(self) -> str:
        """The ``Resource`` identity — the model id (for provenance / eval / logging)."""
        return f"{self.model_id}@{self.revision}" if self.revision else self.model_id

    def _load(self):
        """Lazily build the ONNX session + tokenizer on first use (keeps import cheap)."""
        if self._session is not None:
            return
        with self._load_lock:
            if self._session is not None:  # another thread finished the load while we waited
                return
            import onnxruntime as ort
            from tokenizers import Tokenizer

            tok = Tokenizer.from_file(str(self.tokenizer_path))
            tok.enable_truncation(self.max_length)
            tok.enable_padding()
            session = ort.InferenceSession(str(self.model_path), providers=["CPUExecutionProvider"])
            self._tokenizer = tok
            self._input_names = {i.name for i in session.get_inputs()}
            self._session = session  # published last: a non-None session means everything is ready

    def score(self, query: str, passages: list[str]) -> list[float]:
        """Score each (``query``, passage) pair with one ONNX forward pass over the batch and return
        the relevance logit per passage (the standard single-output cross-encoder convention)."""
        if not passages:
            return []
        self._load()
        import numpy as np

        encs = self._tokenizer.encode_batch([(query, p) for p in passages])  # pair encoding (+[SEP])
        ids = np.array([e.ids for e in encs], dtype=np.int64)
        mask = np.array([e.attention_mask for e in encs], dtype=np.int64)
        feed = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in self._input_names:
            feed["token_type_ids"] = np.array([e.type_ids for e in encs], dtype=np.int64)

        logits = np.asarray(self._session.run(None, feed)[0])  # [B, 1] (regression) or [B, C]
        return logits.reshape(len(passages), -1)[:, 0].astype(float).tolist()
