"""Pipeline stage base classes — pure transforms (no DB / no queue access, D6).

A stage is ``PipelineItem -> Iterator[<results>]``. Transform stages yield
``PipelineItem``s; the terminal embed stage yields ``Embedding``s. The worker calls
``process_batch`` (default maps ``process`` over items; batch-aware stages override
it). ``Pipeline`` is a thin ordered container of stages.
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Iterator, Mapping
from typing import Any, Literal

from tarnrag.contracts import PipelineItem
from bausatz import Component, ComponentFactory


class PipelineStage(Component):
    """
    Base class for a pure transformation stage — a ``Component`` built from a typed ``Config``.
    Subclass a typed helper below (MapperStage / FilterStage) unless a stage needs full control
    (e.g. ``ChunkStage`` drives a ``Chunker`` child directly). Construct from a config, e.g.
    ``CleanAndNormalizeStage(CleanAndNormalizeStage.Config())``; field validation lives on the
    ``Config`` (pydantic), not in a separate ``validate()``.
    """

    class Config(Component.Config):
        """Base stage config. Concrete stages pin ``class_name`` and add their own fields."""

    def __init__(self, config: PipelineStage.Config) -> None:
        super().__init__(config)  # Component.__init__ stores self.config
        # `name` is the stage's identity within a pipeline (DAG node / job.stage_name): an explicit
        # `config.name` wins; otherwise the class tag, which ``PipelineDAG`` suffixes POSITIONALLY
        # (``{tag}-{index}``) when the stage joins a pipeline. Names must be derivable from the
        # pipeline definition alone — a queued job's ``stage_name`` is resolved by the CONSUMER's DAG,
        # in distributed mode a different process, so any process-local state in the name (a global
        # instance counter once lived here) breaks routing whenever the two processes' construction
        # histories differ. Sinks and metrics key off ``tag`` (the type), not on this.
        class_tag = getattr(config, "class_name", None) or type(self).__name__
        self._name: str = config.name or class_tag

    @property
    def name(self) -> str:
        """The stage's identity within a pipeline (DAG node / job target) — explicit ``config.name``,
        else the class tag until ``PipelineDAG`` assigns the positional default (``{tag}-{index}``)."""
        return self._name

    @property
    def tag(self) -> str:
        """The stage's TYPE tag — the Config's ``class_name`` — the sink-registry and obs-metric
        key (falls back to the instance name for un-tagged stages, e.g. test doubles)."""
        return getattr(self.config, "class_name", None) or self._name

    @abstractmethod
    def process(self, item: PipelineItem) -> Iterator[Any]:
        """Transform one item; yield zero or more results."""

    def process_batch(self, items: list[PipelineItem]) -> Iterator[Any]:
        """Batch entrypoint the worker calls (D3). Default maps ``process`` over the
        items one at a time; batch-aware stages (e.g. EmbedStage) override this to
        size their own compute/model calls."""
        for item in items:
            yield from self.process(item)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r})"


class MapperStage(PipelineStage):
    """
    1 -> 1 transform. Override ``map``; upstream metadata is merged automatically.
    """

    @abstractmethod
    def map(self, text: str, metadata: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        """Return ``(new_text, metadata_updates)`` — updates merged onto upstream metadata."""

    def process(self, item: PipelineItem) -> Iterator[PipelineItem]:
        new_text, updates = self.map(item.content, item.metadata)
        yield item.derive(content=new_text, metadata={**item.metadata, **updates})


class FilterStage(PipelineStage):
    """
    1 -> {0, 1}. Override ``should_keep`` (and optionally ``maybe_transform``).
    """

    @abstractmethod
    def should_keep(self, text: str, metadata: dict[str, Any]) -> bool:
        """Return True to keep the item, False to drop it."""

    def maybe_transform(
        self, text: str, metadata: dict[str, Any]
    ) -> tuple[str, dict[str, Any]]:
        """Optionally transform a kept item; default is a no-op."""
        return text, {}

    def process(self, item: PipelineItem) -> Iterator[PipelineItem]:
        if self.should_keep(item.content, item.metadata):
            new_text, updates = self.maybe_transform(item.content, item.metadata)
            yield item.derive(content=new_text, metadata={**item.metadata, **updates})


class Pipeline(Component):
    """
    An ordered list of stages — a container ``Component``. Its ``Config`` lists the stage configs (typed
    ``PipelineStage.Config`` children, so a malformed stage spec — or a stage's malformed child — fails
    when the pipeline spec is validated, not later at build time) and ``_build_children`` builds them
    through the factory, so a whole pipeline is itself a spec:
    ``{"class_name": "pipeline", "stages": [{"class_name": "Chunk", ...}, ...]}``. The
    DAG/orchestrator read ``.stages``; ``run`` threads items through in-process (local testing — the
    distributed engine runs stages individually via the worker). Two alternative constructors:
    ``from_spec`` (a spec dict, via the factory) and ``from_stages`` (pre-made stage instances,
    bypassing the factory — e.g. for tests that inject a fake embedder).
    """

    class Config(Component.Config):
        class_name: Literal["pipeline"] = "pipeline"
        stages: list[PipelineStage.Config]

    config: Pipeline.Config  # narrowed for type checkers (see Component)

    def __init__(self, config: Pipeline.Config) -> None:
        super().__init__(config)  # Component.__init__ stores self.config
        self.stages: list[PipelineStage] = []

    def _build_children(self, factory: ComponentFactory) -> None:
        """Build the (already-validated) stage configs via the SAME factory, asserting each is a
        PipelineStage (``build_as`` does the build + type check)."""
        stages = [factory.build_as(cfg, PipelineStage) for cfg in self.config.stages]
        if not stages:
            raise ValueError("Pipeline must have at least one stage")
        self.stages = stages

    @classmethod
    def from_spec(cls, spec: Mapping[str, Any]) -> Pipeline:
        """Build a Pipeline (and its stages) from a spec —
        ``{"class_name": "pipeline", "stages": [<stage spec>, ...]}`` — via ``ComponentFactory``.
        The factory-driven companion to ``from_stages``; the referenced stage classes must be
        imported (registered) first (importing ``tarnrag.ingestion`` registers the built-ins)."""
        return ComponentFactory.get().create_as(dict(spec), cls)

    @classmethod
    def from_stages(cls, stages: list[PipelineStage]) -> Pipeline:
        """Build a Pipeline directly from already-constructed stages, bypassing the spec/factory —
        for tests and advanced wiring that inject stage internals (e.g. a fake embedder)."""
        if not stages:
            raise ValueError("Pipeline must have at least one stage")
        pipeline = cls(cls.Config(stages=[]))
        pipeline.stages = list(stages)
        return pipeline

    def run(self, items: Iterator[PipelineItem]) -> Iterator[Any]:
        """Thread each input item through every stage in order, yielding final results."""
        for item in items:
            current: list[Any] = [item]
            for stage in self.stages:
                nxt: list[Any] = []
                for it in current:
                    nxt.extend(stage.process(it))
                current = nxt
            yield from current

    def __repr__(self) -> str:
        return "Pipeline([" + ", ".join(repr(s) for s in self.stages) + "])"
