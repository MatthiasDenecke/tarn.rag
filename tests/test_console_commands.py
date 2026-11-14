"""The console's command handlers, REPL loop, report rendering, and ``main`` — driven by a fake ``TarnRag``
whose facade methods return canned ``Outcome``s, with the rich output captured to a string (no store /
embedder / model / real session)."""

import io
import types

import pytest
from rich.console import Console as RichConsole

from tarnrag.console import Console, _load_dotenv, main
from tarnrag.contracts import CorpusStatus, RetrievalResult
from tarnrag.generation.types import GenerationResult
from tarnrag.report import Issue, Outcome, Report, Severity
from tarnrag.retrieval.types import Query, SearchStage, SearchTrace


class FakeTarn:
    """Stand-in for ``TarnRag``: each facade method records its call and returns a preconfigured ``Outcome``
    (or raises, if one is configured as an exception)."""

    def __init__(self, **returns):
        self._returns = returns
        self.calls: list[tuple] = []
        self.settings = types.SimpleNamespace(
            database=types.SimpleNamespace(document_url="sqlite:///corpus.db")
        )

    def _ret(self, name, *args):
        self.calls.append((name, args))
        value = self._returns.get(name)
        if isinstance(value, Exception):
            raise value
        return value

    async def ingest(self, paths):
        return self._ret("ingest", paths)

    async def docs(self):
        return self._ret("docs")

    async def status(self):
        return self._ret("status")

    async def delete(self, document_id):
        return self._ret("delete", document_id)

    async def retrieve(self, query):
        return self._ret("retrieve", query)

    async def explain(self, query):
        return self._ret("explain", query)

    async def ask(self, query):
        return self._ret("ask", query)


def _console(tarn: FakeTarn) -> Console:
    c = Console(tarn)
    c._out = RichConsole(file=io.StringIO(), force_terminal=False, width=200)  # capture output
    return c


def _text(c: Console) -> str:
    return c._out.file.getvalue()


def _result(chunk_id="c1", document_id="doc"):
    return RetrievalResult(
        chunk_id=chunk_id, text="alpha", score=0.9, component_scores={"dense": 0.8},
        document_id=document_id, source_kind="text", standard_id=None, locator=None,
        license_class="public_domain",
    )


# ---------------- ingest ----------------


async def test_ingest_renders_a_status_table():
    statuses = [types.SimpleNamespace(document_id="d1", status="complete", chunk_count=3, embedding_count=3)]
    c = _console(FakeTarn(ingest=Outcome(statuses)))
    await c._do_ingest("a.md b.md")
    assert "d1" in _text(c) and "complete" in _text(c)
    assert c._tarn.calls == [("ingest", (["a.md", "b.md"],))]  # split into paths


async def test_ingest_keeps_a_quoted_path_with_spaces_together():
    """LOW-01: shell-style splitting — ``"my docs/a.pdf"`` is one path, not two."""
    c = _console(FakeTarn(ingest=Outcome([])))
    await c._do_ingest('"my docs/report one.pdf" b.md')
    assert c._tarn.calls == [("ingest", (["my docs/report one.pdf", "b.md"],))]


async def test_ingest_reports_an_unbalanced_quote_instead_of_guessing():
    c = _console(FakeTarn(ingest=Outcome([])))
    await c._do_ingest('"my docs/a.pdf')
    assert "can't read the paths" in _text(c) and c._tarn.calls == []


async def test_ingest_without_arg_prints_usage_and_skips_the_call():
    c = _console(FakeTarn())
    await c._do_ingest("")
    assert "usage: ingest" in _text(c) and c._tarn.calls == []


async def test_ingest_empty_result_says_nothing_ingested():
    c = _console(FakeTarn(ingest=Outcome([])))
    await c._do_ingest("missing")
    assert "nothing ingested" in _text(c)


# ---------------- docs / status ----------------


async def test_docs_lists_documents_with_a_count():
    docs = [types.SimpleNamespace(document_id="d1", chunk_count=2, embedding_count=2)]
    c = _console(FakeTarn(docs=Outcome(docs)))
    await c._do_docs("")
    assert "d1" in _text(c) and "1 document" in _text(c)


async def test_docs_empty_says_no_documents():
    c = _console(FakeTarn(docs=Outcome([])))
    await c._do_docs("")
    assert "no documents" in _text(c)


async def test_status_renders_the_corpus_view():
    status = CorpusStatus(
        document_count=2, chunk_count=4, embedding_count=4, total_chars=100,
        min_chars=10, max_chars=60, mean_chars=50.0, median_chars=50.0, mean_chunks_per_doc=2.0,
    )
    c = _console(FakeTarn(status=Outcome(status)))
    await c._do_status("")
    assert "documents" in _text(c) and "2" in _text(c)


# ---------------- delete ----------------


async def test_delete_reports_deleted():
    c = _console(FakeTarn(delete=Outcome(True)))
    await c._do_delete("d1")
    assert "deleted" in _text(c) and c._tarn.calls == [("delete", ("d1",))]


async def test_delete_reports_missing():
    c = _console(FakeTarn(delete=Outcome(False)))
    await c._do_delete("nope")
    assert "no such document" in _text(c)


async def test_delete_without_arg_prints_usage():
    c = _console(FakeTarn())
    await c._do_delete("")
    assert "usage: delete" in _text(c) and c._tarn.calls == []


# ---------------- retrieve / explain / ask ----------------


async def test_retrieve_renders_results():
    c = _console(FakeTarn(retrieve=Outcome([_result(document_id="tank")])))
    await c._do_retrieve("alpha")
    assert "tank" in _text(c)


async def test_retrieve_empty_says_no_results():
    c = _console(FakeTarn(retrieve=Outcome([])))
    await c._do_retrieve("zzz")
    assert "no results" in _text(c)


async def test_retrieve_without_arg_prints_usage():
    c = _console(FakeTarn())
    await c._do_retrieve("")
    assert "usage: retrieve" in _text(c)


async def test_explain_renders_a_trace():
    trace = SearchTrace(query=Query(text="alpha", top_k=1), stages=[SearchStage("final", [_result()])])
    c = _console(FakeTarn(explain=Outcome(trace)))
    await c._do_explain("alpha")
    assert "alpha" in _text(c)


async def test_explain_empty_trace_says_no_results():
    c = _console(FakeTarn(explain=Outcome(SearchTrace(query=Query(text="z", top_k=1), stages=[]))))
    await c._do_explain("z")
    assert "no results" in _text(c)


async def test_explain_without_arg_prints_usage():
    c = _console(FakeTarn())
    await c._do_explain("")
    assert "usage: explain" in _text(c)


async def test_ask_renders_the_answer():
    c = _console(FakeTarn(ask=Outcome(GenerationResult(answer="forty-two", grounded=True))))
    await c._do_ask("meaning?")
    assert "forty-two" in _text(c)


async def test_ask_without_arg_prints_usage():
    c = _console(FakeTarn())
    await c._do_ask("")
    assert "usage: ask" in _text(c)


async def test_help_lists_the_commands():
    c = _console(FakeTarn())
    await c._do_help("")
    text = _text(c)
    assert "ingest" in text and "retrieve" in text and "ask" in text


# ---------------- report rendering ----------------


async def test_render_report_surfaces_warnings_and_errors():
    report = Report((
        Issue("file not found", Severity.ERROR, subject="bad.md"),
        Issue("partial result", Severity.WARNING),
    ))
    c = _console(FakeTarn(ingest=Outcome([], report)))
    await c._do_ingest("bad.md")
    text = _text(c)
    assert "error:" in text and "bad.md" in text and "file not found" in text
    assert "warning:" in text and "partial result" in text


# ---------------- the REPL loop ----------------


async def test_run_dispatches_commands_then_quits(monkeypatch):
    lines = iter(["help", "bogus", "", "retrieve alpha", "quit"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(lines))
    c = _console(FakeTarn(retrieve=Outcome([_result()])))
    await c.run()
    text = _text(c)
    assert "tarn.rag console" in text and "sqlite:///corpus.db" in text  # banner + the document_url
    assert "ingest" in text  # the help table was printed
    assert "unknown command" in text and "bogus" in text
    assert ("retrieve", ("alpha",)) in c._tarn.calls
    assert "bye" in text


async def test_run_breaks_cleanly_on_eof(monkeypatch):
    def eof(prompt=""):
        raise EOFError

    monkeypatch.setattr("builtins.input", eof)
    c = _console(FakeTarn())
    await c.run()
    assert "bye" in _text(c)


async def test_run_keeps_the_session_alive_on_a_command_error(monkeypatch):
    lines = iter(["ask boom", "quit"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(lines))
    c = _console(FakeTarn(ask=RuntimeError("kaboom")))
    await c.run()
    text = _text(c)
    assert "error:" in text and "kaboom" in text and "bye" in text  # error shown, loop survived to quit


# ---------------- main / dotenv ----------------


def test_main_requires_exactly_one_config_argument():
    with pytest.raises(SystemExit):
        main([])
    with pytest.raises(SystemExit):
        main(["a", "b"])


def test_load_dotenv_is_a_safe_noop():
    _load_dotenv()  # loads a .env if present, no-op without python-dotenv — must not raise


def test_main_loads_dotenv_then_runs_the_session(monkeypatch):
    import tarnrag.console as cm

    calls: list = []
    monkeypatch.setattr(cm, "_load_dotenv", lambda: calls.append("dotenv"))

    async def fake_main(cfg):
        calls.append(("main", cfg))

    monkeypatch.setattr(cm, "_main", fake_main)
    cm.main(["config.json"])
    assert calls == ["dotenv", ("main", "config.json")]  # dotenv loaded, then the session run


async def test_main_session_opens_tarn_and_runs_the_console(monkeypatch):
    import tarnrag.console as cm

    events: list = []

    class _FakeSession:
        def __init__(self, cfg):
            events.append(("open", cfg))

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            events.append("close")
            return False

    async def fake_run(self):
        events.append("run")

    monkeypatch.setattr(cm, "TarnRag", _FakeSession)
    monkeypatch.setattr(cm.Console, "run", fake_run)
    await cm._main("config.json")
    assert events == [("open", "config.json"), "run", "close"]  # opened, ran, closed
