"""Observability interface (metrics + logging) and a no-op implementation.

Core logic must work with ``observability=None`` (every call site guards ``self.obs``);
``NoOpObservability`` is the dev/test stand-in when you want a non-None instance with no
effect. ``StructuredLoggingObservability`` (PP-3) is the first real adapter: one JSON line per
event over stdlib logging — machine-parseable, zero dependencies. Prometheus remains future work.
All adapters live behind this ABC — the domain never imports them. Obs is held by the **worker** (compute metrics) and the
**orchestrator** (lifecycle: enqueue / status / persistence); stages stay pure (D6).
"""

from __future__ import annotations

import json
import logging
import time
from abc import ABC, abstractmethod
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from tarnrag.core.engine.config import ObservabilitySettings


class Observability(ABC):
    """
    Metrics + logging port. ``counter``/``gauge`` are sync (cheap); ``log`` is async.
    """

    @staticmethod
    def create(settings: ObservabilitySettings) -> Observability | None:
        """Build the configured observability adapter, or ``None`` when disabled (core logic guards every
        ``self.obs`` call, so ``None`` = off). Selected by ``settings.type``:
        ``"structured_logging"`` → JSON-lines over stdlib logging (PP-3); anything else (including the
        not-yet-implemented ``"prometheus"``) falls back to the no-op."""
        if not settings.enabled:
            return None
        if settings.type == "structured_logging":
            return StructuredLoggingObservability()
        return NoOpObservability()  # prometheus etc. dispatch here when they land

    @abstractmethod
    async def log(self, level: str, message: str, **context: Any) -> None:
        """Log a message with structured context. Level: debug | info | warning | error."""

    @abstractmethod
    def counter(self, name: str, value: int = 1, tags: dict[str, str] | None = None) -> None:
        """Increment a counter metric."""

    @abstractmethod
    def gauge(self, name: str, value: float, tags: dict[str, str] | None = None) -> None:
        """Set a gauge metric."""

    @contextmanager
    def timer(self, name: str, tags: dict[str, str] | None = None) -> Iterator[None]:
        """Time the wrapped block; records elapsed seconds as ``{name}.seconds`` (gauge).
        Concrete (built on ``gauge``) so every adapter — and the no-op — gets it for free."""
        start = time.perf_counter()
        try:
            yield
        finally:
            self.gauge(f"{name}.seconds", time.perf_counter() - start, tags=tags)


class NoOpObservability(Observability):
    """
    No-op implementation — for development/testing (or when obs is enabled but no real
    adapter is configured yet). All methods do nothing.
    """

    async def log(self, level: str, message: str, **context: Any) -> None:
        pass

    def counter(self, name: str, value: int = 1, tags: dict[str, str] | None = None) -> None:
        pass

    def gauge(self, name: str, value: float, tags: dict[str, str] | None = None) -> None:
        pass


class StructuredLoggingObservability(Observability):
    """JSON-lines adapter (PP-3): every metric and log event becomes ONE machine-parseable JSON line
    on a stdlib logger (``tarnrag.observability``) — greppable in dev, shippable to any log pipeline
    in production, zero new dependencies. Enable with ``OBSERVABILITY__ENABLED=true
    OBSERVABILITY__TYPE=structured_logging``."""

    _LEVELS: ClassVar[dict[str, int]] = {"debug": logging.DEBUG, "info": logging.INFO, "warning": logging.WARNING, "error": logging.ERROR}

    def __init__(self, logger_name: str = "tarnrag.observability") -> None:
        self._logger = logging.getLogger(logger_name)

    def _emit(self, level: int, payload: dict[str, Any]) -> None:
        self._logger.log(level, json.dumps(payload, sort_keys=True, default=str))

    async def log(self, level: str, message: str, **context: Any) -> None:
        self._emit(
            self._LEVELS.get(level, logging.INFO),
            {"event": "log", "level": level, "message": message, **context},
        )

    def counter(self, name: str, value: int = 1, tags: dict[str, str] | None = None) -> None:
        self._emit(logging.INFO, {"event": "counter", "name": name, "value": value, "tags": tags or {}})

    def gauge(self, name: str, value: float, tags: dict[str, str] | None = None) -> None:
        self._emit(logging.INFO, {"event": "gauge", "name": name, "value": value, "tags": tags or {}})
