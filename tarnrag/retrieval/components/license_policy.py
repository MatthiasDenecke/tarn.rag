"""The LicensePolicy seam — which license classes a query's purpose may retrieve (ModusQ §5.6).

A ``LicensePolicy`` is a config-driven ``Component`` (registered ``class_name``, swappable by spec) that
turns a ``Query`` into the ``ChunkFilter`` the retrievers pre-filter with: it owns the **purpose → permitted
``license_class`` set** mapping, and folds in the availability / grounding / method-scope rules that come
from the query itself (delegating those to ``Query.permitted_filter``). The engine builds it from the
``LICENSE_POLICY`` spec in ``Settings.components`` and injects it via the ``RetrievalContext``; deployments
tune the map (or drop in their own policy) without touching the retrievers.

``DefaultLicensePolicy`` ships the ModusQ §5.6 table: every purpose may see the four *shippable* classes —
``customer_licensed`` / ``public_domain`` / ``modusq_authored`` / ``third_party_licensed`` — and
``third_party_copyrighted`` is **never** listed, so it can never be returned (the safety net).
``GENERATION_GROUNDING`` additionally requires ``ai_grounding_allowed`` and all purposes require
``available`` (both carried from ``Query.permitted_filter``).

Grounding is additive: a query that ``grounds_generation`` (what generation retrieves) keeps its caller's
purpose and is permitted the **intersection** of that purpose's classes and the ``GENERATION_GROUNDING``
ones — so generation can only see less than the caller's purpose allows, never more. A purpose absent from
the map permits **nothing** (fail closed): a missing entry can't silently mean "every class", the one
reading under which ``third_party_copyrighted`` would slip through.
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import replace
from typing import Literal

from bausatz import Component
from pydantic import Field

from tarnrag.contracts import ChunkFilter
from tarnrag.retrieval.types import Purpose, Query

# ModusQ §5.6 default: the shippable classes every purpose may see; ``third_party_copyrighted`` is
# deliberately absent from every list, so no purpose can ever return it.
_SHIPPABLE = ("customer_licensed", "public_domain", "modusq_authored", "third_party_licensed")
_DEFAULT_PERMITTED: dict[str, list[str]] = {
    "EXECUTION": list(_SHIPPABLE),
    "AUTHORING": list(_SHIPPABLE),
    "GENERATION_GROUNDING": list(_SHIPPABLE),
}


class LicensePolicy(Component):
    """Port: build the permitted-chunk ``ChunkFilter`` for a query (its purpose → license classes, plus the
    query's availability / grounding / scope). Concrete policies pin ``class_name``."""

    class Config(Component.Config):
        """Base license-policy config; concrete policies pin ``class_name``."""

    @abstractmethod
    def filter_for(self, query: Query) -> ChunkFilter:
        """The permitted-chunk filter the retrievers pre-filter with for ``query``."""


class DefaultLicensePolicy(LicensePolicy):
    """The ModusQ §5.6 default policy: map each purpose to its permitted ``license_class`` set (the four
    shippable classes by default; ``third_party_copyrighted`` is never permitted), then fold in the query's
    availability / grounding / scope. The ``permitted`` map is config — a deployment tunes it per purpose;
    an absent purpose permits nothing, and a query that grounds generation gets its purpose's classes
    intersected with ``GENERATION_GROUNDING``'s (see the module docstring)."""

    class Config(LicensePolicy.Config):
        class_name: Literal["default_license"] = "default_license"
        # purpose value (e.g. "EXECUTION") -> permitted license_class list. A purpose absent from the map
        # permits no class at all (fail closed).
        permitted: dict[str, list[str]] = Field(
            default_factory=lambda: {k: list(v) for k, v in _DEFAULT_PERMITTED.items()}
        )

    config: DefaultLicensePolicy.Config

    def filter_for(self, query: Query) -> ChunkFilter:
        purposes = [query.purpose]
        if query.grounds_generation and query.purpose is not Purpose.GENERATION_GROUNDING:
            purposes.append(Purpose.GENERATION_GROUNDING)  # tighten: both purposes must permit a class
        classes = self._classes(purposes[0])
        for purpose in purposes[1:]:
            allowed = set(self._classes(purpose))
            classes = [c for c in classes if c in allowed]
        # permitted_filter() owns available / grounding / scope; the policy adds the license-class set.
        return replace(query.permitted_filter(), license_classes=tuple(classes))

    def _classes(self, purpose: Purpose) -> list[str]:
        """The classes ``purpose`` may retrieve — none when the map doesn't list it (fail closed)."""
        return list(self.config.permitted.get(purpose.value, []))
