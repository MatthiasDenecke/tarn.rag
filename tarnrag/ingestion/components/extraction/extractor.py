"""Extractor seam — a ``Source`` becomes a ``StructuredDocument``.

Extractors are config-driven ``Component``s (registered by ``class_name``, like the pipeline stages),
so the document-processing pipeline (extract → enrich) is configurable as data and a user can drop in
their own extractor. ``extract`` routes a ``Source`` to the right extractor by ``source_kind`` — an
explicit ``metadata['extractor']`` override wins, else the per-kind default, else plain text (graceful
degradation). Tiered fast/high-fidelity routing slots in here as the route values grow.

See ``doc/extraction-seam-design.md``.
"""

from __future__ import annotations

from abc import abstractmethod
from pathlib import Path
from typing import Any

from bausatz import Component
from pydantic import BaseModel, Field

from tarnrag.contracts import Element, StructuredDocument
from tarnrag.core.hashing import compute_content_hash


class Source(BaseModel):
    """An extractor's input: where the bytes/text are, the document identity, and routing hints."""

    source_id: str
    source_kind: str = ""  # "pdf" | "markdown" | "text" | … (detected/declared by the caller)
    path: str | None = None  # filesystem path, if loading from disk
    content: str | None = None  # already-loaded text, if not from disk
    metadata: dict[str, Any] = Field(default_factory=dict)  # hints, e.g. {"extractor": "plain_text"}


class Extractor(Component):
    """Port: produce a ``StructuredDocument`` from a ``Source``. Concrete extractors pin ``class_name``."""

    class Config(Component.Config):
        """Base extractor config; concrete extractors pin ``class_name`` and add their own options."""

    @abstractmethod
    def extract(self, source: Source) -> StructuredDocument:
        """Parse the source into a structured document."""

    @staticmethod
    def _create_document(
        source: Source, text: str, elements: list[Element], *, extractor: str, default_kind: str
    ) -> StructuredDocument:
        """Assemble the ``StructuredDocument`` (the shared tail of every ``extract``): identity from the
        source, ``source_kind`` falling back to ``default_kind``, and the computed content hash."""
        return StructuredDocument(
            source_id=source.source_id,
            source_kind=source.source_kind or default_kind,
            extractor=extractor,
            text=text,
            elements=elements,
            content_hash=compute_content_hash(text),
        )

    @staticmethod
    def _read_text(source: Source) -> str:
        """The source's text: inline ``content`` if given, else read ``path`` (utf-8), else empty."""
        if source.content is not None:
            return source.content
        if source.path:
            return Path(source.path).read_text(encoding="utf-8", errors="replace")
        return ""
