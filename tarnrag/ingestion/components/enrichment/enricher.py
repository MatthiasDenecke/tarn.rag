"""The Enricher component family: annotate a ``StructuredDocument`` after extraction (FR-5).

An ``Enricher`` is a pure ``Component`` that appends typed ``Annotation``s to a document's elements (or
the document itself) — entities, topics, classifications, … — each recording its ``producer`` and a
deterministic-vs-generative flag (anti-hallucination). The ``EnrichStage`` runs a configured, ordered
list of them over ``item.document``; the annotations then flow into chunks (the chunker aggregates
element annotations into ``ChunkProvenance``). Users register their own (NER/topic) behind this port on
the same framework — no base-class edits (AC-2).
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any

from bausatz import Component

from tarnrag.contracts import Annotation, Element, Geometry, StructuredDocument


class Enricher(Component):
    """Port: annotate a ``StructuredDocument`` in place. Subclasses call ``annotate`` / ``annotate_span``
    rather than touching ``.annotations`` directly — so the ``producer`` is recorded automatically and
    the element's internals stay hidden. Runs after extraction, before chunking."""

    class Config(Component.Config):
        """Base enricher config; concrete enrichers pin ``class_name`` and add their own options."""

    @abstractmethod
    def enrich(self, document: StructuredDocument) -> None:
        """Append annotations to the document and/or its elements (mutates in place)."""

    def annotate(
        self,
        target: Element | StructuredDocument,
        type: str,
        value: dict[str, Any],
        *,
        span: Geometry | None = None,
        deterministic: bool = True,
    ) -> None:
        """Record an annotation by this enricher on an element OR the document (both expose
        ``annotations``); ``producer`` is filled from this enricher, ``span`` is an optional sub-region."""
        target.annotations.append(
            Annotation(
                producer=self.config.class_name,
                type=type,
                value=value,
                span=span,
                deterministic=deterministic,
            )
        )

    def annotate_span(
        self,
        element: Element,
        type: str,
        value: dict[str, Any],
        start: int,
        end: int,
        *,
        deterministic: bool = True,
    ) -> None:
        """Annotate an element over its local range ``[start, end)`` — maps to a document span via
        ``Element.subspan``."""
        self.annotate(element, type, value, span=[element.subspan(start, end)], deterministic=deterministic)
