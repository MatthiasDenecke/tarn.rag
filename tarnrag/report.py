"""Structured, user-facing reports for the TarnRag facade.

Every facade call returns an :class:`Outcome` — its ``value`` plus a :class:`Report` of any
:class:`Issue`\\ s encountered alongside it. The report is *empty* (and falsy) when all went well, so
there is nothing to show; non-fatal problems — a path that wasn't found, a partial failure — are recorded
here rather than printed or silently skipped, and every UI over the facade (the console, the tiqtasq REST
API) decides how to surface them. Fatal failures still raise; the report is for issues the call carried on
*past*.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Severity(str, Enum):
    """How bad an issue is. ``WARNING`` — the call carried on without it; ``ERROR`` — part of the request
    could not be honoured (but the call still returned a value). A ``str`` enum so it serialises cleanly
    for the REST layer."""

    WARNING = "warning"
    ERROR = "error"


@dataclass(frozen=True)
class Issue:
    """One problem encountered during a facade call, phrased for a user. ``subject`` is what the issue is
    about (a path, a document id); ``message`` says what happened."""

    message: str
    severity: Severity = Severity.WARNING
    subject: str | None = None


@dataclass(frozen=True)
class Report:
    """The issues a single facade call encountered — empty (and falsy) when all went well, so callers can
    test ``if outcome.report:`` before rendering anything."""

    issues: tuple[Issue, ...] = ()

    def __bool__(self) -> bool:
        return bool(self.issues)

    @property
    def ok(self) -> bool:
        """True when nothing went wrong (the report is empty)."""
        return not self.issues


@dataclass(frozen=True)
class Outcome[T]:
    """A facade call's result: the ``value`` it produced, plus a ``report`` of any issues encountered
    alongside it (empty when all went well).

    ``understood_as`` is set only by ``retrieve`` / ``explain`` / ``ask`` when conversation memory
    rewrote the question: it is the standalone question the call actually ran (what a UI shows as
    "understood as"). It is ``None`` when the question ran as typed, when memory is off, and for every
    other call. It is part of the result rather than the report, because a rewrite is not an issue."""

    value: T
    report: Report = field(default_factory=Report)
    understood_as: str | None = None
