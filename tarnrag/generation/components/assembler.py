"""The EvidenceAssembler seam — turn a Reasoner's cited indices into provenance-bearing Citations.

A pure ``Component``: it maps each reasoning step's cited evidence indices to ``Citation``s (chunk id +
locator + the geometry / header-path that makes the span highlightable), producing the proof tree. The
default ``ProvenanceAssembler`` reads geometry straight off each hit's ``ChunkProvenance`` via
``Citation.from_result``. Richer assemblers (dedup citations, merge adjacent spans, add table-cell
citations) plug in behind this seam.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Literal

from bausatz import Component

from tarnrag.generation.components.reasoner import ReasonedAnswer
from tarnrag.generation.types import Citation, ProofStep


class EvidenceAssembler(Component):
    """Port: build the proof tree (``list[ProofStep]``) from a ``ReasonedAnswer``."""

    class Config(Component.Config):
        """Base assembler config; concrete assemblers pin ``class_name``."""

    @abstractmethod
    def assemble(self, reasoned: ReasonedAnswer, grounded: list[bool] | None = None) -> list[ProofStep]:
        """Map each step's cited evidence to ``Citation``s, producing the inspectable proof tree.
        ``grounded`` (aligned to ``reasoned.steps``) stamps each step's verdict; ``None`` ⇒ all grounded."""


class ProvenanceAssembler(EvidenceAssembler):
    """Build each ``ProofStep``'s citations straight from the cited hits' layout provenance."""

    class Config(EvidenceAssembler.Config):
        class_name: Literal["provenance"] = "provenance"

    config: ProvenanceAssembler.Config

    def assemble(self, reasoned: ReasonedAnswer, grounded: list[bool] | None = None) -> list[ProofStep]:
        return [
            ProofStep(
                claim=step.claim,
                citations=[Citation.from_result(reasoned.evidence[i]) for i in step.cited],
                grounded=grounded[idx] if grounded is not None else True,
            )
            for idx, step in enumerate(reasoned.steps)
        ]
