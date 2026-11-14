"""Conversation memory through the ``TarnRag`` facade (doc/conversation-memory-design.md).

Model-free: the ``hash`` embedder ingests and retrieves without ONNX, and a scripted LLM stands in for
both the rewriter and the reader — it records every prompt, so a test can assert exactly what reached
the model. That is the point of these tests: with memory OFF the facade behaves as it always did, and
with memory ON the *standalone* question — not the follow-up as typed — is what retrieval and the
reader receive, while the conversation itself never reaches the reader.
"""

import asyncio
import json
from pathlib import Path

import pytest
from sqlalchemy import func, select

from tarnrag import ConversationTurn, Severity, TarnRag
from tarnrag.core.engine.config import (
    CONVERSATION_MEMORY,
    GENERATION_PIPELINE,
    DatabaseSettings,
    EmbeddingSettings,
    Settings,
)
from tarnrag.core.resources.llm import Completion, LanguageModel, Prompt
from tarnrag.storage.repository import DocumentRepository

FIRST = "Which frame is the TX-200 built on?"
FOLLOW_UP = "Which oil grade does it require?"
STANDALONE = "Which oil grade does the Cooper-Bessemer GMV frame require?"


class ScriptedLLM(LanguageModel):
    """Stands in for the rewriter, the reader and the fact-checker, told apart by their system prompts.
    Records every prompt by role. ``rewrite`` is the rewriter's raw reply (so a test can make it
    unusable); ``grounded`` is the fact-checker's verdict for every claim."""

    def __init__(self, rewrite: str | None = None, *, answer: str = "an answer", grounded: bool = True):
        self.rewrite = json.dumps({"standalone_question": STANDALONE}) if rewrite is None else rewrite
        self.answer = answer
        self.grounded = grounded
        self.rewrites: list[Prompt] = []
        self.reads: list[Prompt] = []
        self.fail_reads = False

    def identity(self) -> str:
        return "scripted"

    async def complete(self, prompt: Prompt) -> Completion:
        system = prompt.system or ""
        if "rewrite the LATEST question" in system:
            self.rewrites.append(prompt)
            return Completion(text=self.rewrite)
        if "fact-checker" in system:
            return Completion(text=json.dumps({"verdicts": [self.grounded] * 8}))
        self.reads.append(prompt)
        if self.fail_reads:
            raise ConnectionError("reader endpoint unreachable")
        return Completion(
            text=json.dumps({"answer": self.answer, "steps": [{"claim": self.answer, "cited": [1]}]})
        )

    def questions_read(self) -> list[str]:
        """The ``Question:`` line of each reader prompt — what the reader was actually asked."""
        return [p.user.split("\n", 1)[0].removeprefix("Question: ") for p in self.reads]


def _settings(tmp_path: Path, *, memory: dict | None = None, generation: dict | None = None) -> Settings:
    """Embedded + hash embedder + a one-call reader. ``memory`` is merged into an ENABLED
    conversation_memory spec; ``None`` leaves the library default (memory off)."""
    components: dict = {
        GENERATION_PIPELINE: {
            "class_name": "generation_pipeline",
            "reasoner": {"class_name": "single_hop", "top_k": 2},
            **(generation or {}),
        }
    }
    if memory is not None:
        components[CONVERSATION_MEMORY] = {"class_name": "conversation_memory", "enabled": True, **memory}
    return Settings(
        _env_file=None,
        MODE="embedded",
        EMBEDDING_DIMENSION=8,
        ID_POLICY="caller",
        database=DatabaseSettings(document_url=f"sqlite:///{tmp_path}/store.db"),
        embedding=EmbeddingSettings(provider="hash"),
        components=components,
    )


async def _ingest(tarn: TarnRag, tmp_path: Path) -> None:
    src = tmp_path / "corpus"
    src.mkdir(exist_ok=True)
    (src / "models.txt").write_text("The TX-200 is built on a Cooper-Bessemer GMV frame.")
    (src / "lube.txt").write_text("Cooper-Bessemer GMV frames require oil of grade ISO VG 68.")
    assert (await tarn.ingest([str(src)])).report.ok


async def _stored_turns(tarn: TarnRag) -> int:
    """Every row in ``conversation_turns``, across all conversations — straight from the table."""
    table = tarn._repository.conversation_turns_table
    async with tarn._repository.engine.connect() as conn:
        return (await conn.execute(select(func.count()).select_from(table))).scalar_one()


# ---------------- memory OFF: today's behaviour, untouched ----------------


async def test_memory_off_by_default_every_call_is_stateless(tmp_path):
    llm = ScriptedLLM()
    async with TarnRag(_settings(tmp_path), llm=llm) as tarn:
        await _ingest(tarn, tmp_path)
        assert tarn.session_id is None
        await tarn.ask(FIRST)
        outcome = await tarn.ask(FOLLOW_UP)
        await tarn.retrieve(FOLLOW_UP)
        await tarn.explain(FOLLOW_UP)

        assert outcome.report.ok
        assert llm.rewrites == []  # no rewrite call, ever
        assert llm.questions_read() == [FIRST, FOLLOW_UP]  # the follow-up reaches the reader as typed
        assert await _stored_turns(tarn) == 0  # nothing remembered


async def test_memory_off_the_query_is_exactly_the_stateless_one(tmp_path, monkeypatch):
    """Off must be the old code path: the Query handed to retrieval carries the text as typed, the
    caller's top_k, the default over-fetch, and no annotation."""
    async with TarnRag(_settings(tmp_path), llm=ScriptedLLM()) as tarn:
        await _ingest(tarn, tmp_path)
        seen = []
        search = tarn._retrieval.search

        async def spy(query):
            seen.append(query)
            return await search(query)

        monkeypatch.setattr(tarn._retrieval, "search", spy)
        await tarn.retrieve(FOLLOW_UP, top_k=3)
        (query,) = seen
        assert (query.text, query.top_k, query.dense_k, query.annotations) == (FOLLOW_UP, 3, 50, [])


async def test_memory_off_the_conversation_methods_say_so(tmp_path):
    """Not a silent no-op: each returns an empty value plus a WARNING that memory is disabled."""
    async with TarnRag(_settings(tmp_path), llm=ScriptedLLM(), session_id="ignored") as tarn:
        assert tarn.session_id is None  # a requested session is ignored while memory is off
        for outcome, empty in (
            (await tarn.history(), []),
            (await tarn.forget(), 0),
            (await tarn.new_conversation(), None),
        ):
            assert outcome.value == empty
            (issue,) = outcome.report.issues
            assert issue.severity is Severity.WARNING and "disabled" in issue.message


async def test_an_explicitly_disabled_spec_is_off(tmp_path):
    settings = _settings(tmp_path, memory={"enabled": False, "session_id": "shift-a"})
    llm = ScriptedLLM()
    async with TarnRag(settings, llm=llm) as tarn:
        await _ingest(tarn, tmp_path)
        await tarn.ask(FIRST)
        await tarn.ask(FOLLOW_UP)
        assert tarn.session_id is None and llm.rewrites == [] and await _stored_turns(tarn) == 0


# ---------------- memory ON: the follow-up is resolved ----------------


async def test_memory_on_the_follow_up_reaches_the_reader_rewritten(tmp_path):
    llm = ScriptedLLM(answer="Cooper-Bessemer GMV")
    async with TarnRag(_settings(tmp_path, memory={}), llm=llm) as tarn:
        await _ingest(tarn, tmp_path)
        await tarn.ask(FIRST)
        assert llm.rewrites == []  # the first turn has nothing to resolve against → no extra call

        outcome = await tarn.ask(FOLLOW_UP)
        assert outcome.report.ok
        assert len(llm.rewrites) == 1  # exactly one extra call for the follow-up
        assert "Cooper-Bessemer GMV" in llm.rewrites[0].user  # turn 1's answer reached the rewriter
        assert llm.rewrites[0].user.rstrip().endswith(f"Latest question: {FOLLOW_UP}")
        assert llm.questions_read() == [FIRST, STANDALONE]  # the reader is asked the STANDALONE question


async def test_the_conversation_never_reaches_the_reader(tmp_path):
    """Grounding is not weakened: the reader sees the standalone question and passages — no transcript,
    no earlier answer. History decides which question is asked, never what counts as evidence."""
    llm = ScriptedLLM(answer="a-distinctive-earlier-answer")
    async with TarnRag(_settings(tmp_path, memory={}), llm=llm) as tarn:
        await _ingest(tarn, tmp_path)
        await tarn.ask(FIRST)
        await tarn.ask(FOLLOW_UP)
        follow_up_read = llm.reads[1]
        assert "a-distinctive-earlier-answer" not in follow_up_read.user
        assert FIRST not in follow_up_read.user and "Conversation so far" not in follow_up_read.user


async def test_memory_on_retrieval_searches_the_standalone_question(tmp_path, monkeypatch):
    async with TarnRag(_settings(tmp_path, memory={}), llm=ScriptedLLM()) as tarn:
        await _ingest(tarn, tmp_path)
        await tarn.retrieve(FIRST)
        seen = []
        search = tarn._retrieval.search

        async def spy(query):
            seen.append(query)
            return await search(query)

        monkeypatch.setattr(tarn._retrieval, "search", spy)
        outcome = await tarn.retrieve(FOLLOW_UP, top_k=2)
        (query,) = seen
        assert query.text == STANDALONE and query.top_k == 2
        (annotation,) = query.annotations  # the rewrite travels with the query, flagged as generated
        assert annotation.type == "contextualized" and annotation.deterministic is False
        assert annotation.value == {"original": FOLLOW_UP, "turns_used": 1}
        assert len(outcome.value) == 2 and outcome.report.ok


async def test_history_shows_each_turn_as_typed_and_as_understood(tmp_path):
    llm = ScriptedLLM(answer="ISO VG 68")
    async with TarnRag(_settings(tmp_path, memory={}), llm=llm) as tarn:
        await _ingest(tarn, tmp_path)
        await tarn.ask(FIRST)
        await tarn.ask(FOLLOW_UP)
        first, second = (await tarn.history()).value
        assert isinstance(second, ConversationTurn)
        assert (first.question, first.standalone_question) == (FIRST, FIRST)  # nothing to rewrite
        assert (second.question, second.standalone_question) == (FOLLOW_UP, STANDALONE)
        assert second.kind == "ask" and second.answer == "ISO VG 68"
        assert second.grounded is None and second.abstained is False  # no grounding check ran: unchecked
        assert (first.ordinal, second.ordinal) == (0, 1) and second.session_id == tarn.session_id
        assert [t.question for t in (await tarn.history(last=1)).value] == [FOLLOW_UP]


async def test_an_ungrounded_refused_answer_is_remembered_with_its_flags(tmp_path):
    """Every ask turn stores its answer — a refusal included — with the verdicts that say what it was."""
    generation = {"grounding_checker": {"class_name": "llm_grounding"}, "abstain": True, "refusal": "No."}
    llm = ScriptedLLM(grounded=False)
    async with TarnRag(_settings(tmp_path, memory={}, generation=generation), llm=llm) as tarn:
        await _ingest(tarn, tmp_path)
        result = (await tarn.ask(FIRST)).value
        assert result.abstained is True
        (turn,) = (await tarn.history()).value
        assert (turn.answer, turn.grounded, turn.abstained) == ("No.", False, True)
        await tarn.ask(FOLLOW_UP)
        assert "A (the assistant declined to answer): No." in llm.rewrites[0].user  # marked for the rewriter


# ---------------- who records ----------------


async def test_a_checked_answer_is_remembered_grounded_and_an_unchecked_one_is_not(tmp_path):
    """Without a grounding check, ``GenerationResult.grounded`` is True by default, not by verification —
    so the turn records ``None`` (unchecked), and the rewriter is told so instead of shown it as reliable."""
    checked = {"grounding_checker": {"class_name": "llm_grounding"}}
    async with TarnRag(_settings(tmp_path / "checked", memory={}, generation=checked), llm=ScriptedLLM()) as tarn:
        (tmp_path / "checked").mkdir(exist_ok=True)
        await _ingest(tarn, tmp_path / "checked")
        await tarn.ask(FIRST)
        (turn,) = (await tarn.history()).value
        assert turn.grounded is True  # verified
    llm = ScriptedLLM()
    async with TarnRag(_settings(tmp_path, memory={}), llm=llm) as tarn:
        await _ingest(tarn, tmp_path)
        await tarn.ask(FIRST)
        (turn,) = (await tarn.history()).value
        assert turn.grounded is None  # nothing verified it
        await tarn.ask(FOLLOW_UP)
        (rewrite,) = llm.rewrites
        assert "A (unchecked" in rewrite.user


async def test_retrieve_records_a_question_only_turn_that_a_later_ask_can_refer_to(tmp_path):
    llm = ScriptedLLM()
    async with TarnRag(_settings(tmp_path, memory={}), llm=llm) as tarn:
        await _ingest(tarn, tmp_path)
        await tarn.retrieve("TX-200 frame")
        (turn,) = (await tarn.history()).value
        assert turn.kind == "retrieve" and turn.question == "TX-200 frame"
        assert turn.answer is None and turn.grounded is None and turn.abstained is None

        await tarn.ask(FOLLOW_UP)
        assert "[1] Q: TX-200 frame" in llm.rewrites[0].user  # the search was remembered
        assert "no answer was recorded" in llm.rewrites[0].user


async def test_explain_is_a_dry_run_it_reads_memory_but_records_nothing(tmp_path):
    async with TarnRag(_settings(tmp_path, memory={}), llm=ScriptedLLM()) as tarn:
        await _ingest(tarn, tmp_path)
        await tarn.ask(FIRST)
        trace = (await tarn.explain(FOLLOW_UP)).value
        assert trace.query.text == STANDALONE  # it shows what retrieve WOULD search…
        assert trace.query.annotations[0].value["original"] == FOLLOW_UP  # …and what was typed
        assert trace.results  # the trace is a real retrieval over the rewritten query
        assert [t.question for t in (await tarn.history()).value] == [FIRST]  # …without changing state
        await tarn.explain(FOLLOW_UP)  # repeatable: same input, same conversation
        assert len((await tarn.history()).value) == 1


async def test_an_operation_that_fails_records_nothing(tmp_path):
    llm = ScriptedLLM()
    async with TarnRag(_settings(tmp_path, memory={}), llm=llm) as tarn:
        await _ingest(tarn, tmp_path)
        llm.fail_reads = True
        with pytest.raises(ConnectionError):
            await tarn.ask(FIRST)
        assert (await tarn.history()).value == []  # a turn that produced no answer is not a turn


# ---------------- sessions ----------------


async def test_each_open_starts_a_fresh_conversation_by_default(tmp_path):
    settings = _settings(tmp_path, memory={})
    llm = ScriptedLLM()
    async with TarnRag(settings, llm=llm) as first:
        await _ingest(first, tmp_path)
        await first.ask(FIRST)
        first_session = first.session_id
    async with TarnRag(settings, llm=llm) as second:
        assert second.session_id and second.session_id != first_session
        assert (await second.history()).value == []
        await second.ask(FOLLOW_UP)
        assert llm.rewrites == []  # the earlier conversation is not inherited
        assert await _stored_turns(second) == 2  # …but it is still in the store (keep all)


async def test_a_session_id_passed_to_the_facade_resumes_that_conversation(tmp_path):
    settings = _settings(tmp_path, memory={})
    llm = ScriptedLLM(answer="Cooper-Bessemer GMV")
    async with TarnRag(settings, llm=llm, session_id="shift-a") as first:
        await _ingest(first, tmp_path)
        await first.ask(FIRST)
    async with TarnRag(settings, llm=llm, session_id="shift-a") as second:  # a new process, same store
        assert second.session_id == "shift-a"
        await second.ask(FOLLOW_UP)
        assert len(llm.rewrites) == 1 and "Cooper-Bessemer GMV" in llm.rewrites[0].user
        assert [t.ordinal for t in (await second.history()).value] == [0, 1]


async def test_a_configured_session_id_resumes_and_the_argument_overrides_it(tmp_path):
    settings = _settings(tmp_path, memory={"session_id": "from-config"})
    async with TarnRag(settings, llm=ScriptedLLM()) as tarn:
        assert tarn.session_id == "from-config"
        await _ingest(tarn, tmp_path)
        await tarn.ask(FIRST)
    async with TarnRag(settings, llm=ScriptedLLM()) as tarn:
        assert [t.question for t in (await tarn.history()).value] == [FIRST]  # resumed from the config
    async with TarnRag(settings, llm=ScriptedLLM(), session_id="from-argument") as tarn:
        assert tarn.session_id == "from-argument" and (await tarn.history()).value == []


async def test_new_conversation_starts_over_but_keeps_the_old_transcript(tmp_path):
    llm = ScriptedLLM()
    async with TarnRag(_settings(tmp_path, memory={}), llm=llm) as tarn:
        await _ingest(tarn, tmp_path)
        await tarn.ask(FIRST)
        old = tarn.session_id
        outcome = await tarn.new_conversation()
        assert outcome.value == tarn.session_id != old and outcome.report.ok
        await tarn.ask(FOLLOW_UP)
        assert llm.rewrites == []  # the new conversation does not see the old turn
        assert [t.question for t in (await tarn.history()).value] == [FOLLOW_UP]
        assert await tarn._repository.conversation_turns(old)  # the old transcript is still stored


async def test_forget_deletes_the_conversations_turns(tmp_path):
    llm = ScriptedLLM()
    async with TarnRag(_settings(tmp_path, memory={}), llm=llm) as tarn:
        await _ingest(tarn, tmp_path)
        await tarn.ask(FIRST)
        await tarn.retrieve(FIRST)
        session = tarn.session_id
        assert (await tarn.forget()).value == 2
        assert tarn.session_id == session  # same conversation, nothing remembered
        assert (await tarn.history()).value == [] and await _stored_turns(tarn) == 0
        rewrites_before = len(llm.rewrites)  # (the second turn above was itself a follow-up)
        await tarn.ask(FOLLOW_UP)
        assert len(llm.rewrites) == rewrites_before  # nothing left to resolve against → no rewrite call


async def test_the_window_bounds_what_the_rewriter_sees_not_what_is_stored(tmp_path):
    llm = ScriptedLLM()
    async with TarnRag(_settings(tmp_path, memory={"max_turns": 2}), llm=llm) as tarn:
        await _ingest(tarn, tmp_path)
        for i in range(4):
            await tarn.retrieve(f"question {i}")
        shown = llm.rewrites[-1].user  # what the rewriter saw for "question 3": turns 0-2 exist, 2 fit
        assert "question 0" not in shown and "[1] Q: " in shown and "[3] Q:" not in shown
        assert len((await tarn.history()).value) == 4


# ---------------- a rewrite that could not be made is reported ----------------


async def test_an_unusable_rewrite_falls_back_and_is_reported(tmp_path):
    llm = ScriptedLLM(rewrite="I believe they mean the GMV frame.")  # prose, not the JSON contract
    async with TarnRag(_settings(tmp_path, memory={}), llm=llm) as tarn:
        await _ingest(tarn, tmp_path)
        await tarn.ask(FIRST)
        outcome = await tarn.ask(FOLLOW_UP)
        assert llm.questions_read() == [FIRST, FOLLOW_UP]  # carried on with the question as typed…
        (issue,) = outcome.report.issues  # …and said so
        assert issue.severity is Severity.WARNING and issue.subject == FOLLOW_UP
        assert "used as typed" in issue.message
        turn = (await tarn.history()).value[-1]
        assert turn.standalone_question == FOLLOW_UP  # recorded as it was actually executed


async def test_a_follow_up_without_an_llm_is_searched_as_typed_and_reported(tmp_path, monkeypatch):
    """A retrieval-only session needs no key — until a follow-up needs resolving. Then the search still
    runs (as typed) and the report names the missing key."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    async with TarnRag(_settings(tmp_path, memory={})) as tarn:  # no injected LLM, no key
        await _ingest(tarn, tmp_path)
        assert (await tarn.retrieve(FIRST)).report.ok  # the first turn needs no LLM at all
        outcome = await tarn.retrieve(FOLLOW_UP)
        assert outcome.value  # results came back
        (issue,) = outcome.report.issues
        assert issue.severity is Severity.WARNING and "ANTHROPIC_API_KEY" in issue.message
        with pytest.raises(RuntimeError, match="needs an LLM key"):  # ask still fails fast, as before…
            await tarn.ask(FOLLOW_UP)
        assert len((await tarn.history()).value) == 2  # …and that failed ask recorded nothing


async def test_a_failing_rewrite_call_propagates_and_records_nothing(tmp_path):
    class BrokenRewriter(ScriptedLLM):
        async def complete(self, prompt):
            if "rewrite the LATEST question" in (prompt.system or ""):
                raise ConnectionError("rewriter endpoint unreachable")
            return await super().complete(prompt)

    async with TarnRag(_settings(tmp_path, memory={}), llm=BrokenRewriter()) as tarn:
        await _ingest(tarn, tmp_path)
        await tarn.retrieve(FIRST)
        with pytest.raises(ConnectionError):
            await tarn.retrieve(FOLLOW_UP)
        assert len((await tarn.history()).value) == 1


# ---------------- wiring ----------------


class _GatedLLM(ScriptedLLM):
    """A ``ScriptedLLM`` whose reader waits to be released, so a test can act while an ``ask`` is
    mid-generation."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.reading = asyncio.Event()
        self.release = asyncio.Event()

    async def complete(self, prompt: Prompt) -> Completion:
        system = prompt.system or ""
        if "rewrite the LATEST question" not in system and "fact-checker" not in system:
            self.reading.set()
            await self.release.wait()
        return await super().complete(prompt)


@pytest.mark.parametrize("switch", ["new_conversation", "forget"])
async def test_a_session_switch_waits_for_the_turn_in_flight(tmp_path, switch):
    """``new_conversation`` / ``forget`` while an ``ask`` is generating: the turn is still recorded in
    the conversation it was resolved against, and the switch applies after it — a fresh conversation
    starts empty, and a forgotten one stays empty."""
    llm = _GatedLLM()
    async with TarnRag(_settings(tmp_path, memory={}), llm=llm) as tarn:
        await _ingest(tarn, tmp_path)
        started_in = tarn.session_id
        asking = asyncio.create_task(tarn.ask(FIRST))
        await llm.reading.wait()  # the ask is mid-generation
        switching = asyncio.create_task(getattr(tarn, switch)())
        for _ in range(5):
            await asyncio.sleep(0)
        assert not switching.done()  # it waits for the turn in flight
        llm.release.set()
        await asyncio.gather(asking, switching)
        recorded = await tarn._repository.conversation_turns(started_in)
        if switch == "new_conversation":
            assert [t.question for t in recorded] == [FIRST]  # recorded where it started
            assert tarn.session_id != started_in and (await tarn.history()).value == []
        else:
            assert recorded == []  # forgotten after it was recorded, not resurrected by it


async def test_concurrent_calls_on_one_facade_take_turns(tmp_path):
    """Two ``ask``s at once on one memory-enabled facade run one after the other: the second is resolved
    against the first's recorded turn, and both are recorded, in order."""
    llm = ScriptedLLM()
    async with TarnRag(_settings(tmp_path, memory={}), llm=llm) as tarn:
        await _ingest(tarn, tmp_path)
        await asyncio.gather(tarn.ask(FIRST), tarn.ask(FOLLOW_UP))
        first, second = (await tarn.history()).value
        assert (first.question, second.question) == (FIRST, FOLLOW_UP)
        assert second.standalone_question == STANDALONE and len(llm.rewrites) == 1  # it saw the first turn


async def test_memory_off_calls_are_not_serialized(tmp_path):
    """Without memory there is no conversation to keep in order, so nothing is locked."""
    llm = _GatedLLM()
    async with TarnRag(_settings(tmp_path), llm=llm) as tarn:
        await _ingest(tarn, tmp_path)
        asking = asyncio.create_task(tarn.ask(FIRST))
        await llm.reading.wait()
        assert (await tarn.retrieve(FOLLOW_UP)).value  # runs while the ask is still generating
        llm.release.set()
        await asking


@pytest.mark.parametrize("operation", ["ask", "retrieve"])
async def test_a_failed_memory_write_returns_the_result_with_a_warning(tmp_path, monkeypatch, operation):
    """The result is already computed when the turn is recorded, so a store error there (a lock, a lost
    connection) is reported, not raised — the optional memory never costs the caller the answer."""
    async with TarnRag(_settings(tmp_path, memory={}), llm=ScriptedLLM(answer="GMV")) as tarn:
        await _ingest(tarn, tmp_path)

        async def locked(turn):
            raise RuntimeError("database is locked")

        monkeypatch.setattr(tarn._repository, "append_turn", locked)
        outcome = await getattr(tarn, operation)(FIRST)
        assert outcome.value  # the answer / the passages
        (issue,) = outcome.report.issues
        assert issue.severity is Severity.WARNING and "could not record this turn" in issue.message
        assert "database is locked" in issue.message
        assert (await tarn.history()).value == []


async def test_only_a_rewrite_builds_the_llm(tmp_path, monkeypatch):
    """Reading and recording turns, a first turn, ``history`` and ``forget`` never build the session's
    LLM; the first follow-up that needs a rewrite builds it, once."""
    settings = _settings(tmp_path, memory={})
    settings.llm.api_key = "k"
    async with TarnRag(settings) as tarn:
        await _ingest(tarn, tmp_path)
        built: list[LanguageModel] = []

        def build(llm_settings):
            built.append(ScriptedLLM())
            return built[-1]

        monkeypatch.setattr(LanguageModel, "create", staticmethod(build))
        assert (await tarn.retrieve(FIRST)).report.ok  # nothing to resolve yet
        await tarn.history()
        assert built == []
        await tarn.retrieve(FOLLOW_UP)
        await tarn.retrieve("and how often is it changed?")
        assert len(built) == 1 and len(built[0].rewrites) == 2  # built once, by the first rewrite
        await tarn.forget()
        assert len(built) == 1


def test_history_refuses_a_negative_count(tmp_path):
    with pytest.raises(ValueError, match="last"):
        asyncio.run(TarnRag(_settings(tmp_path, memory={})).history(last=-1))


async def test_a_bad_memory_spec_fails_at_open_and_releases_the_store(tmp_path):
    settings = _settings(tmp_path, memory={"max_turns": 0})
    tarn = TarnRag(settings)
    disconnected = []
    real_disconnect = DocumentRepository.disconnect

    async def disconnect(self):
        disconnected.append(self)
        await real_disconnect(self)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(DocumentRepository, "disconnect", disconnect)
        with pytest.raises(ValueError, match="max_turns"):
            await tarn.open()
    assert disconnected == [tarn._repository]  # the store open() connected was released


async def test_the_llm_built_for_the_rewrite_is_closed_by_the_facade(tmp_path):
    """A retrieve-only session with memory on may build the LLM just for the rewrite; the facade that
    built it releases it (an injected one is left to its owner — covered in test_tarnrag.py)."""
    settings = _settings(tmp_path, memory={})
    settings.llm.api_key = "k"  # lets the facade construct a backend (lazily connecting — no network)
    tarn = await TarnRag(settings).open()
    context = tarn._conversation_context()
    assert tarn._llm is None  # a context alone builds nothing
    assert context.language_model() is tarn._llm is not None  # built once, on first need
    assert tarn._generation is None  # generation was never needed
    closed: list[str] = []

    class _Closable:
        async def aclose(self) -> None:
            closed.append("llm")

    tarn._llm = _Closable()
    await tarn.close()
    assert closed == ["llm"]


async def test_generation_and_the_rewrite_share_one_llm(tmp_path):
    llm = ScriptedLLM()
    async with TarnRag(_settings(tmp_path, memory={}), llm=llm) as tarn:
        assert tarn._gen().llm is tarn._conversation_context().language_model() is llm


async def test_the_switch_is_read_from_a_yaml_config(tmp_path):
    def config(enabled: str) -> Path:
        path = tmp_path / f"memory-{enabled}.yaml"
        path.write_text(
            "MODE: embedded\n"
            "EMBEDDING_DIMENSION: 8\n"
            "ID_POLICY: caller\n"
            "embedding: { provider: hash }\n"
            f"database: {{ document_url: 'sqlite:///{tmp_path}/yaml-{enabled}.db' }}\n"
            "components:\n"
            "  generation_pipeline:\n"
            "    class_name: generation_pipeline\n"
            "    reasoner: { class_name: single_hop, top_k: 2 }\n"
            "  conversation_memory:\n"
            "    class_name: conversation_memory\n"
            f"    enabled: {enabled}          # <- the on/off switch\n"
            "    max_turns: 6\n"
            "    max_answer_chars: 600\n"
            "    contextualizer: { class_name: condense, max_tokens: 256 }\n"
        )
        return path

    for enabled, expected_reads in (("true", [FIRST, STANDALONE]), ("false", [FIRST, FOLLOW_UP])):
        llm = ScriptedLLM()
        async with TarnRag(config(enabled), llm=llm) as tarn:
            await _ingest(tarn, tmp_path)
            await tarn.ask(FIRST)
            await tarn.ask(FOLLOW_UP)
            assert (tarn.session_id is not None) is (enabled == "true")
            assert llm.questions_read() == expected_reads
