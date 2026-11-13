"""The console's command handlers, REPL loop, report rendering, and ``main`` — driven by a fake ``TarnRag``
whose facade methods return canned ``Outcome``s, with the rich output captured to a string (no store /
embedder / model / real session)."""

import io
import types

import pytest
from rich.console import Console as RichConsole

from tarnrag.console import Console, _load_dotenv, main
from tarnrag.contracts import ConversationTurn, CorpusStatus, RetrievalResult
from tarnrag.generation.types import GenerationResult
from tarnrag.report import Issue, Outcome, Report, Severity
from tarnrag.retrieval.types import Query, SearchStage, SearchTrace


class FakeTarn:
    """Stand-in for ``TarnRag``: each facade method records its call and returns a preconfigured ``Outcome``
    (or raises, if one is configured as an exception). ``session_id`` is ``None`` (conversation memory
    off) unless given — then ``history`` / ``new_conversation`` / ``forget`` are served the same way."""

    def __init__(self, session_id=None, **returns):
        self._returns = returns
        self.calls: list[tuple] = []
        self.session_id = session_id
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

    async def history(self, last=None):
        return self._ret("history", last)

    async def new_conversation(self):
        return self._ret("new_conversation")

    async def forget(self):
        return self._ret("forget")


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
    assert "history" in text and "new" in text and "forget" in text  # the conversation commands


# ---------------- conversation memory ----------------

_MEMORY_OFF = Report((Issue("conversation memory is disabled — set …enabled: true", Severity.WARNING),))


def _turn(question, standalone=None, *, kind="ask", ordinal=0, **fields):
    if kind == "ask":
        fields = {"answer": "ISO VG 68", "grounded": True, "abstained": False, **fields}
    return ConversationTurn(
        session_id="s1", kind=kind, question=question,
        standalone_question=standalone or question, ordinal=ordinal, **fields,
    )


async def test_ask_shows_what_a_follow_up_was_understood_as():
    """A rewrite is never hidden: after the answer comes back, the console reads the turn just recorded
    and prints the standalone question that was actually asked."""
    recorded = _turn("which oil does it need?", "Which oil does the GMV frame need?", ordinal=1)
    tarn = FakeTarn(
        session_id="s1",
        ask=Outcome(GenerationResult(answer="ISO VG 68", proof=[], evidence=[])),
        history=Outcome([recorded]),
    )
    c = _console(tarn)
    await c._do_ask("which oil does it need?")
    text = _text(c)
    assert "understood as:" in text and "Which oil does the GMV frame need?" in text
    assert text.index("understood as:") < text.index("ISO VG 68")  # the interpretation precedes the answer
    assert ("history", (1,)) in tarn.calls  # one cheap read of the latest turn


async def test_retrieve_shows_what_a_follow_up_was_understood_as():
    recorded = _turn("and its seals?", "pump seal replacement", kind="retrieve")
    tarn = FakeTarn(session_id="s1", retrieve=Outcome([_result()]), history=Outcome([recorded]))
    c = _console(tarn)
    await c._do_retrieve("and its seals?")
    assert "understood as:" in _text(c) and "pump seal replacement" in _text(c)


async def test_a_question_used_as_typed_prints_no_understood_line():
    tarn = FakeTarn(
        session_id="s1",
        ask=Outcome(GenerationResult(answer="GMV", proof=[], evidence=[])),
        history=Outcome([_turn("which frame is the TX-200 built on?")]),
    )
    c = _console(tarn)
    await c._do_ask("which frame is the TX-200 built on?")
    assert "understood as" not in _text(c)


async def test_with_memory_off_ask_and_retrieve_never_read_history():
    tarn = FakeTarn(
        ask=Outcome(GenerationResult(answer="x", proof=[], evidence=[])), retrieve=Outcome([_result()])
    )
    c = _console(tarn)
    await c._do_ask("q")
    await c._do_retrieve("q")
    assert [name for name, _ in tarn.calls] == ["ask", "retrieve"]  # no history call
    assert "understood as" not in _text(c)


async def test_history_renders_each_turn_as_typed_as_understood_and_its_answer():
    turns = [
        _turn("which frame is the TX-200 built on?", answer="Cooper-Bessemer GMV"),
        _turn("which oil does it need?", "Which oil does the GMV frame need?", ordinal=1),
        _turn("pump seals", kind="retrieve", ordinal=2),
        _turn("warranty?", ordinal=3, answer="Not enough evidence.", grounded=False, abstained=True),
        _turn("flow rate?", ordinal=4, answer="about 40 l/min", grounded=False),
    ]
    c = _console(FakeTarn(session_id="s1", history=Outcome(turns)))
    await c._do_history("")
    text = _text(c)
    assert "which oil does it need?" in text and "Which oil does the GMV frame need?" in text
    assert "✓ Cooper-Bessemer GMV" in text  # a grounded answer
    assert "(search only)" in text  # the retrieve turn has no answer
    assert "abstained" in text and "Not enough evidence." in text
    assert "✗ about 40 l/min" in text  # answered, but not fully grounded
    assert "5 turn(s)" in text and "s1" in text


async def test_history_of_a_new_conversation_says_nothing_is_remembered():
    c = _console(FakeTarn(session_id="s1", history=Outcome([])))
    await c._do_history("")
    assert "nothing remembered yet" in _text(c)


async def test_new_starts_a_fresh_conversation_and_says_the_transcript_is_kept():
    c = _console(FakeTarn(session_id="s1", new_conversation=Outcome("s2")))
    await c._do_new("")
    text = _text(c)
    assert "new conversation" in text and "s2" in text and "kept" in text


async def test_forget_reports_how_many_turns_were_deleted():
    c = _console(FakeTarn(session_id="s1", forget=Outcome(3)))
    await c._do_forget("")
    assert "forgot" in _text(c) and "3 turn(s)" in _text(c)


async def test_the_conversation_commands_explain_themselves_when_memory_is_off():
    """Not a silent no-op: each prints the facade's warning that memory is disabled — and nothing that
    reads like success."""
    tarn = FakeTarn(
        history=Outcome([], _MEMORY_OFF), new_conversation=Outcome(None, _MEMORY_OFF),
        forget=Outcome(0, _MEMORY_OFF),
    )
    for command in ("_do_history", "_do_new", "_do_forget"):
        c = _console(tarn)
        await getattr(c, command)("")
        text = _text(c)
        assert "warning:" in text and "disabled" in text
        assert "forgot" not in text and "new conversation" not in text and "nothing remembered" not in text


async def test_the_banner_states_whether_memory_is_on(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda prompt="": "quit")
    on, off = _console(FakeTarn(session_id="abc123")), _console(FakeTarn())
    await on.run()
    await off.run()
    assert "conversation memory: on" in _text(on) and "abc123" in _text(on)
    assert "conversation memory: off" in _text(off)


async def test_run_dispatches_the_conversation_commands(monkeypatch):
    lines = iter(["history", "new", "forget", "quit"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(lines))
    tarn = FakeTarn(
        session_id="s1", history=Outcome([]), new_conversation=Outcome("s2"), forget=Outcome(0)
    )
    await _console(tarn).run()
    assert [name for name, _ in tarn.calls] == ["history", "new_conversation", "forget"]


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
