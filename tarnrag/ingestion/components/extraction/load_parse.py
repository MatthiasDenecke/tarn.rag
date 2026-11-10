"""LoadAndParseStage — the structured-extraction stage: a source becomes a ``StructuredDocument``.

It is a **container component**: its extractors are *children*, built through the one framework factory
in ``_build_children`` (exactly the way ``Pipeline`` builds its stage children), so the whole tree —
Pipeline → LoadAndParse → extractors — is assembled in one recursive pass from ``Settings.components``,
with no per-stage instantiation. ``Config.routes`` maps a ``source_kind`` to an extractor component
spec (``class_name`` + options), so choosing e.g. Docling for PDFs — with its options — is a config
edit; an unrouted kind falls back to ``default_extractor``. A per-document ``metadata['extractor']``
override is the one runtime choice a construction-time build can't pre-make, so it is built on demand
(once, shared with the routes).

The stage sets ``item.content = document.text`` (so the existing text path keeps working) and
``item.document`` (which the enrichers and the structure-aware chunker consume next).
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Literal

from bausatz import ComponentFactory
from pydantic import Field

from tarnrag.contracts import PipelineItem
from tarnrag.ingestion.components.extraction.extractor import Extractor, Source
from tarnrag.ingestion.components.extraction.html import HtmlExtractor
from tarnrag.ingestion.components.extraction.markdown import MarkdownExtractor
from tarnrag.ingestion.components.extraction.pdf import PdfTextExtractor
from tarnrag.ingestion.components.extraction.plain_text import PlainTextExtractor
from tarnrag.ingestion.pipeline.pipeline import PipelineStage

# Default source_kind -> extractor Config. Config-driven: override per format via Settings.components
# (a route is any extractor spec — ``{"class_name": "docling", ...}`` for high-fidelity PDFs).
_DEFAULT_ROUTES: dict[str, type[Extractor.Config]] = {
    "text": PlainTextExtractor.Config,
    "txt": PlainTextExtractor.Config,
    "markdown": MarkdownExtractor.Config,
    "md": MarkdownExtractor.Config,
    "html": HtmlExtractor.Config,
    "htm": HtmlExtractor.Config,
    "pdf": PdfTextExtractor.Config,  # the fast tier
}


def infer_source_kind(path: str | None) -> str:
    """The routing key for a file path: its lower-cased extension (``txt`` / ``md`` / ``pdf`` / ``html`` /
    …), or ``""`` when there's no path or extension. The **single** extension→kind parser — used both by
    ``LoadAndParseStage`` (to route) and by ``IngestionEngine`` (to stamp ``metadata['source_type']``), so
    the two can't drift to different vocabularies."""
    return Path(path).suffix.lower().lstrip(".") if path else ""


class LoadAndParseStage(PipelineStage):
    """Load + structured-parse a document, setting ``item.document`` + ``item.content = document.text``."""

    class Config(PipelineStage.Config):
        class_name: Literal["LoadAndParse"] = "LoadAndParse"
        # source_kind -> extractor child (a spec's class_name + options, validated into its Config);
        # the config-driven route map.
        routes: dict[str, Extractor.Config] = Field(
            default_factory=lambda: {kind: config() for kind, config in _DEFAULT_ROUTES.items()}
        )
        # fallback when the source_kind isn't routed.
        default_extractor: Extractor.Config = Field(default_factory=PlainTextExtractor.Config)

    config: LoadAndParseStage.Config

    def __init__(self, config: LoadAndParseStage.Config) -> None:
        super().__init__(config)
        self._built: dict[str, Extractor] = {}  # canonical spec -> instance (one per distinct spec)
        self._by_kind: dict[str, Extractor] = {}  # source_kind -> its (shared) extractor child
        self._default: Extractor | None = None

    def _build_children(self, factory: ComponentFactory) -> None:
        """Build the route extractors as children, through the framework factory (the same hook
        ``Pipeline`` uses for its stages) — so they join the one recursive tree build."""
        self._built = {}
        self._by_kind = {kind: self._extractor(cfg, factory) for kind, cfg in self.config.routes.items()}
        self._default = self._extractor(self.config.default_extractor, factory)

    def process(self, item: PipelineItem) -> Iterator[PipelineItem]:
        self.ensure_built()
        md = item.metadata
        path = md.get("source_path")
        source_kind = infer_source_kind(path) or md.get("source_type") or ""
        document = self._select(source_kind, md.get("extractor")).extract(
            Source(
                source_id=md.get("source_id") or md.get("doc_id") or str(uuid.uuid4()),
                source_kind=source_kind,
                path=path,
                content=None if path else item.content,
                metadata=md,
            )
        )
        yield PipelineItem(
            content=document.text,
            metadata={
                **md,
                "doc_id": md.get("doc_id") or document.source_id,
                "loaded": True,
                # The extractor-confirmed kind, under the key the repository persists
                # (``_DOC_PROVENANCE``'s ``source_kind`` column). Distinct from ``source_type``, the
                # caller's INPUT hint this stage routes on — stamping the output key here is what makes
                # ``documents.source_kind`` real (it used to silently fall back to ``"document"``).
                "source_kind": document.source_kind,
            },
            document=document,
            provenance=item.provenance,
        )

    def _select(self, source_kind: str, override: str | None) -> Extractor:
        """The configured route's extractor (a prebuilt child), or — for the per-document override —
        the named extractor, built once on demand and shared with the routes."""
        if override:
            factory = ComponentFactory.get()
            return self._extractor(factory.validate({"class_name": override}), factory)
        return self._by_kind.get(source_kind, self._default)

    def _extractor(self, config: Extractor.Config, factory: ComponentFactory) -> Extractor:
        """Build (or reuse) the extractor for ``config`` via the factory; dedups identical configs so
        several routes share one instance."""
        key = json.dumps(config.model_dump(), sort_keys=True)
        if key not in self._built:
            self._built[key] = factory.build_as(config, Extractor)
        return self._built[key]
