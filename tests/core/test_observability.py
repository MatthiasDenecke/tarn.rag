"""Observability ABC + NoOpObservability."""

from tarnrag.core.engine.observability import NoOpObservability, Observability


async def test_noop_methods_are_callable_and_do_nothing():
    obs = NoOpObservability()
    obs.counter("a")
    obs.counter("a", 5, tags={"k": "v"})
    obs.gauge("g", 1.5)
    await obs.log("info", "hello", x=1)  # async, no-op


class _RecordingGauges(NoOpObservability):
    """No-op except it records gauge calls — to prove the concrete timer emits one."""

    def __init__(self):
        self.gauges: list[tuple[str, float]] = []

    def gauge(self, name, value, tags=None):
        self.gauges.append((name, value))


def test_timer_emits_a_seconds_gauge():
    obs = _RecordingGauges()
    with obs.timer("stage.Embed.process"):
        pass
    assert len(obs.gauges) == 1
    name, value = obs.gauges[0]
    assert name == "stage.Embed.process.seconds"
    assert value >= 0.0


def test_timer_is_inherited_concrete_on_the_abc():
    # timer is defined on the ABC (not abstract), so every adapter gets it for free.
    assert "timer" not in Observability.__abstractmethods__
    assert Observability.__abstractmethods__ == {"log", "counter", "gauge"}


def test_create_returns_none_when_disabled_else_the_configured_adapter():
    """The factory: disabled -> None (core logic guards on None); enabled -> the configured adapter
    (only the no-op ships today)."""
    from tarnrag.core.engine.config import ObservabilitySettings

    assert Observability.create(ObservabilitySettings(enabled=False)) is None
    assert isinstance(Observability.create(ObservabilitySettings(enabled=True)), NoOpObservability)


# ---------------- the structured-logging adapter (PP-3) ----------------


async def test_structured_logging_emits_one_json_line_per_event(caplog):
    import json
    import logging

    from tarnrag.core.engine.observability import StructuredLoggingObservability

    obs = StructuredLoggingObservability()
    with caplog.at_level(logging.INFO, logger="tarnrag.observability"):
        obs.counter("ingest.documents", 3, tags={"mode": "embedded"})
        obs.gauge("stage.Embed.seconds", 1.25)
        await obs.log("warning", "slow stage", stage="Embed", ms=1250)
    events = [json.loads(r.message) for r in caplog.records]
    assert events[0] == {"event": "counter", "name": "ingest.documents", "value": 3, "tags": {"mode": "embedded"}}
    assert events[1] == {"event": "gauge", "name": "stage.Embed.seconds", "value": 1.25, "tags": {}}
    assert events[2] == {"event": "log", "level": "warning", "message": "slow stage", "stage": "Embed", "ms": 1250}
    assert caplog.records[2].levelno == logging.WARNING  # level mapped onto the stdlib logger


async def test_structured_logging_timer_and_unknown_level(caplog):
    import json
    import logging

    from tarnrag.core.engine.observability import StructuredLoggingObservability

    obs = StructuredLoggingObservability()
    with caplog.at_level(logging.INFO, logger="tarnrag.observability"):
        with obs.timer("stage.Chunk.process"):
            pass
        await obs.log("bogus-level", "still logged")  # unknown level -> INFO, never dropped
    timer_event = json.loads(caplog.records[0].message)
    assert timer_event["event"] == "gauge" and timer_event["name"] == "stage.Chunk.process.seconds"
    assert caplog.records[1].levelno == logging.INFO


def test_create_dispatches_structured_logging():
    from tarnrag.core.engine.config import ObservabilitySettings
    from tarnrag.core.engine.observability import (
        Observability,
        StructuredLoggingObservability,
    )

    obs = Observability.create(ObservabilitySettings(enabled=True, type="structured_logging"))
    assert isinstance(obs, StructuredLoggingObservability)
