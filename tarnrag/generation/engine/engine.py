"""GenerationEngine — question → answer + proof tree + evidence (the Goal-3 facade).

The library entry point, sibling of ``IngestionEngine`` / ``RetrievalEngine``. ``create()`` wires the
composition root: it builds the retrieval engine (consumed through the ``RetrievalEngineProtocol`` port),
the ``LanguageModel`` resource (``LanguageModel.create`` over ``Settings.llm``), and the
``GenerationPipeline`` from the ``GENERATION_PIPELINE`` spec — then delegates ``answer`` to the pipeline
over a ``GenerationContext``. tiqtasq.backend wraps this in REST. Building a concrete ``RetrievalEngine``
here is the allowed direction (``generation → retrieval``); the pipeline + seams only ever see the port.
"""

from __future__ import annotations

from typing import Self

from bausatz import ComponentFactory

from tarnrag.core.engine.config import GENERATION_PIPELINE, Settings, get_settings
from tarnrag.core.resources.llm import LanguageModel
from tarnrag.generation.context import GenerationContext
from tarnrag.generation.pipeline.pipeline import GenerationPipeline
from tarnrag.generation.types import GenerationResult
from tarnrag.retrieval.engine.engine import RetrievalEngine
from tarnrag.retrieval.engine.retrieval_engine_protocol import RetrievalEngineProtocol
from tarnrag.retrieval.types import Query


class GenerationEngine:
    """Async question facade: retrieve → read → proof tree. Built via ``create()`` from ``Settings``, or
    constructed directly with an injected retrieval port + LLM (tests / custom wiring)."""

    def __init__(
        self,
        retrieval: RetrievalEngineProtocol,
        llm: LanguageModel,
        pipeline: GenerationPipeline,
        *,
        owns_resources: bool = False,
    ) -> None:
        self.retrieval = retrieval
        self.llm = llm
        self._pipeline = pipeline
        # Whether aclose() releases the retrieval port + LLM: True only on the standalone ``create``
        # path, which builds them itself. Injected resources (a composition root's — TarnRag's shared
        # retrieval engine and LLM) are closed by their owner, never here.
        self._owns_resources = owns_resources

    @classmethod
    def assemble(
        cls,
        retrieval: RetrievalEngineProtocol,
        llm: LanguageModel,
        settings: Settings,
        *,
        owns_resources: bool = False,
    ) -> GenerationEngine:
        """Build the engine over a pre-built retrieval port + LLM, with the ``GenerationPipeline`` from
        ``Settings`` (``Settings`` guarantees the spec is present). The shared wiring used by both
        ``create`` (full standalone — which passes ``owns_resources=True``) and a composition root
        (``TarnRag``) that injects an already-open retrieval engine it keeps ownership of."""
        pipeline = ComponentFactory.get().create_as(
            settings.components[GENERATION_PIPELINE], GenerationPipeline
        )
        return cls(retrieval, llm, pipeline, owns_resources=owns_resources)

    @classmethod
    async def create(cls, settings: Settings | None = None) -> GenerationEngine:
        """Wire the composition standalone: build the retrieval engine (as the port) + the LLM, then
        :meth:`assemble`. The engine owns what it built, so ``aclose`` releases both."""
        settings = settings or get_settings()
        retrieval = await RetrievalEngine.create(settings)
        llm = LanguageModel.create(settings.llm)
        return cls.assemble(retrieval, llm, settings, owns_resources=True)

    async def answer(self, query: Query) -> GenerationResult:
        """Answer a question: retrieve, read, and return the answer + proof tree + evidence. Retrieval is
        marked ``grounds_generation`` (the pipeline enforces it, on top of the query's own purpose), so
        chunks ingested with ``ai_grounding_allowed=False`` never reach the reader, and the license classes
        are never wider than the query's purpose allows."""
        return await self._pipeline.answer(query, GenerationContext(self.retrieval, self.llm))

    async def answer_text(self, text: str) -> GenerationResult:
        """Convenience over :meth:`answer`: build a :class:`Query` from a raw question string."""
        return await self.answer(Query(text=text))

    async def aclose(self) -> None:
        """Release the retrieval engine + LLM **when this engine built them** (the standalone ``create``
        path); an engine assembled over injected resources closes nothing — their owner (the composition
        root) does. Best-effort on the retrieval port (a fake may expose no ``aclose``)."""
        if not self._owns_resources:
            return
        await self.llm.aclose()
        closer = getattr(self.retrieval, "aclose", None)
        if closer is not None:
            await closer()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()
