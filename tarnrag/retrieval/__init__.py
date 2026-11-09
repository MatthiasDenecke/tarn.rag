"""Retrieval domain — public surface.

``RetrievalEngine`` is the high-level facade (``RetrievalEngine.create()``); ``Query`` /
``RetrievalResult`` are its input/output. The retrieval methods are config-driven Components: the engine
builds a ``Searcher`` from the spec — a ``RetrievalPipeline`` composing ``Retriever`` (dense / sparse),
``Fuser`` (identity / rrf), the optional ``Merger`` (auto-merging) and ``Reranker`` (cross-encoder), or a
``RoutingRetrievalPipeline`` that runs a ``QueryClassifier`` and dispatches to a per-type sub-pipeline.
These are the seams users plug into; the repository, embedder, and cross-encoder are internal.
"""

from tarnrag.contracts import MethodRef, RetrievalResult
from tarnrag.retrieval.components.classifier import (
    GenericQueryClassifier,
    IntentQueryClassifier,
    QueryClassifier,
    StructuralQueryClassifier,
)
from tarnrag.retrieval.components.fuser import Fuser
from tarnrag.retrieval.components.license_policy import (
    DefaultLicensePolicy,
    LicensePolicy,
)
from tarnrag.retrieval.components.merger import AutoMerger, Merger
from tarnrag.retrieval.components.reranker import (
    CrossEncoderReranker,
    LlmJudgeReranker,
    Reranker,
)
from tarnrag.retrieval.components.retriever import (
    HydeRetriever,
    MultiQueryRetriever,
    RetrievalContext,
    Retriever,
)
from tarnrag.retrieval.engine.engine import RetrievalEngine, RetrievalError
from tarnrag.retrieval.engine.retrieval_engine_protocol import RetrievalEngineProtocol
from tarnrag.retrieval.pipeline.pipeline import RetrievalPipeline
from tarnrag.retrieval.pipeline.router import RoutingRetrievalPipeline
from tarnrag.retrieval.pipeline.searcher import Searcher
from tarnrag.retrieval.types import Query, RetrieverCandidates, SearchStage, SearchTrace

__all__ = [
    "AutoMerger",
    "CrossEncoderReranker",
    "DefaultLicensePolicy",
    "Fuser",
    "GenericQueryClassifier",
    "HydeRetriever",
    "IntentQueryClassifier",
    "LicensePolicy",
    "LlmJudgeReranker",
    "Merger",
    "MethodRef",
    "MultiQueryRetriever",
    "Query",
    "QueryClassifier",
    "Reranker",
    "RetrievalContext",
    "RetrievalEngine",
    "RetrievalEngineProtocol",
    "RetrievalError",
    "RetrievalPipeline",
    "RetrievalResult",
    "Retriever",
    "RetrieverCandidates",
    "RoutingRetrievalPipeline",
    "SearchStage",
    "SearchTrace",
    "Searcher",
    "StructuralQueryClassifier",
]
