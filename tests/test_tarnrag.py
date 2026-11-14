"""The TarnRag facade.

Two tiers. The **plumbing** tests — construction, ``open`` / ``close`` / ``retrieval_context``, the
index-identity stamp+validate, and the report paths — run everywhere: they never embed, so a dummy
tokenizer file (see ``_plumbing_settings``) is enough and no ONNX model is needed. The **end-to-end**
tests that actually embed text (ingest → retrieve → ask) are marked ``@requires_model`` and skip when the
model hasn't been fetched. The LLM is always injected, so no API key is needed.
"""

import json
from pathlib import Path

import pytest

from examples.common import MODEL_DIR, corpus
from tarnrag.core.engine.config import DatabaseSettings, EmbeddingSettings, Settings
from tarnrag.core.exceptions import IngestionError
from tarnrag.core.resources.llm import StaticLanguageModel
from tarnrag.report import Severity
from tarnrag.tarnrag import TarnRag

requires_model = pytest.mark.skipif(
    not (MODEL_DIR / "model.onnx").exists(), reason="model not fetched (scripts/fetch_model.py)"
)


# ---------------- model-free plumbing (runs in CI) ----------------


def _fake_model_dir(tmp_path: Path) -> Path:
    """A model dir holding only a dummy ``tokenizer.json`` — enough for the embedder's identity /
    fingerprint (which hashes that file), with no real model. The facade plumbing never embeds, so no
    ``model.onnx`` / ``onnxruntime`` is needed and these tests run anywhere."""
    model_dir = tmp_path / "model"
    model_dir.mkdir(exist_ok=True)
    (model_dir / "tokenizer.json").write_text("{}")  # any bytes — only its sha256 is read
    return model_dir


def _plumbing_settings(tmp_path: Path, **embedding: object) -> Settings:
    """Minimal embedded settings over a temp store + the dummy model dir. ``embedding`` overrides feed
    ``EmbeddingSettings`` (e.g. a different ``query_prefix`` to get a distinct embedding fingerprint)."""
    return Settings(
        _env_file=None,
        MODE="embedded",
        EMBEDDING_DIMENSION=8,
        database=DatabaseSettings(document_url=f"sqlite:///{tmp_path}/plumbing.db"),
        embedding=EmbeddingSettings(model_dir=str(_fake_model_dir(tmp_path)), **embedding),
    )


def test_init_loads_settings_from_a_path(tmp_path):
    """``__init__`` with a path (str / Path) loads the JSON config into ``settings`` — no engines opened."""
    cfg = tmp_path / "c.json"
    cfg.write_text(json.dumps({"EMBEDDING_DIMENSION": 384, "database": {"document_url": "sqlite:///x.db"}}))
    settings = TarnRag(cfg).settings  # Path
    assert settings.EMBEDDING_DIMENSION == 384 and "x.db" in settings.database.document_url
    assert TarnRag(str(cfg)).settings.EMBEDDING_DIMENSION == 384  # str


def test_init_loads_settings_from_a_yaml_path(tmp_path):
    """A YAML config loads the same way a JSON one does — the loader dispatches on the extension. YAML's
    comments are why the example configs use it; the parsed mapping is identical to the JSON form."""
    cfg = tmp_path / "c.yaml"
    cfg.write_text(
        "# a commented config\n"
        "EMBEDDING_DIMENSION: 384\n"
        "database:\n"
        "  document_url: sqlite:///x.db  # the §8 index store\n"
    )
    settings = TarnRag(cfg).settings
    assert settings.EMBEDDING_DIMENSION == 384 and "x.db" in settings.database.document_url


def test_from_file_rejects_an_unknown_extension(tmp_path):
    """An unsupported extension fails loudly (not silently as JSON) — the message names the accepted set."""
    cfg = tmp_path / "c.toml"
    cfg.write_text("EMBEDDING_DIMENSION = 384")
    with pytest.raises(ValueError, match="unsupported config extension"):
        Settings.from_file(cfg)


async def test_open_wires_shared_resources_then_close_releases(tmp_path):
    """The composition root, end to end without a model: ``__init__`` leaves the engines unbuilt; the
    context manager's ``open`` builds the repository + embedder once and the retrieval engine over them;
    ingestion is lazy (P6); ``retrieval_context`` exposes the shared ``(store, embedder)``; ``close``
    releases it."""
    tarn = TarnRag(_plumbing_settings(tmp_path))
    assert tarn._repository is None and tarn._ingestion is None  # __init__: nothing built yet

    async with tarn:  # __aenter__ -> open()
        # retrieval is eager and shares the one repository + embedder (no duplication, no reach-around)
        assert tarn._repository is tarn._retrieval.repository
        assert tarn._embedder is tarn._retrieval.embedder
        # P6: ingestion is NOT spun up by a read/open — it builds lazily, over the same store
        assert tarn._ingestion is None
        ingestion = await tarn._ingestion_engine()
        assert ingestion is tarn._ingestion and ingestion.repository is tarn._repository
        ctx = tarn.retrieval_context()
        assert ctx.store is tarn._repository and ctx.embedder is tarn._embedder
    # __aexit__ -> close() disconnected the store


async def test_open_refuses_an_index_built_with_a_different_embedder(tmp_path):
    """P5/P4: the index identity is stamped once and then validated, never re-stamped on open — so opening
    a store whose index was built with a different embedding config is refused, not silently masked. No
    model needed: the mismatch is caught at open (before any embedding), and open() cleans up on the way
    out."""
    # First open stamps the identity (default config).
    async with TarnRag(_plumbing_settings(tmp_path)):
        pass
    # Reopen the SAME store with a different embedding fingerprint (same dim + tokenizer, new query prefix).
    with pytest.raises(IngestionError, match="different embedding pipeline"):
        async with TarnRag(_plumbing_settings(tmp_path, query_prefix="query: ")):
            pass


async def test_ingest_missing_paths_and_empty_store_queries(tmp_path):
    """The output-free facade methods over an empty store, no embedding: ``ingest`` resolves paths up
    front and reports each one that matched nothing (never silently skipped); ``docs`` / ``delete`` work
    on an empty store."""
    async with TarnRag(_plumbing_settings(tmp_path)) as tarn:
        outcome = await tarn.ingest(["nope-a.txt", "nope-b.txt"])  # nothing resolves -> no embedding
        assert outcome.value == []
        assert {i.subject for i in outcome.report.issues} == {"nope-a.txt", "nope-b.txt"}
        assert all(i.severity is Severity.WARNING for i in outcome.report.issues)

        assert (await tarn.docs()).value == []
        assert (await tarn.delete("nope")).value is False


def _caller_hash_settings(tmp_path: Path) -> Settings:
    """Embedded settings with ``ID_POLICY='caller'`` (stems are document ids) + the hash embedder, so the
    ingest-behavior tests actually ingest without a model."""
    return Settings(
        _env_file=None,
        MODE="embedded",
        EMBEDDING_DIMENSION=8,
        ID_POLICY="caller",
        database=DatabaseSettings(document_url=f"sqlite:///{tmp_path}/store.db"),
        embedding=EmbeddingSettings(provider="hash"),
    )


async def test_ingest_reports_stem_collisions_instead_of_overwriting(tmp_path):
    """Under ``ID_POLICY='caller'`` the document id is the file stem, so ``a.md`` + ``a.txt`` share the id
    ``'a'`` — the second must not silently upsert over the first: it is skipped with an ERROR issue naming
    both files, and exactly one document lands. The hash embedder keeps this model-free."""
    src = tmp_path / "in"
    src.mkdir()
    (src / "a.md").write_text("alpha as markdown")
    (src / "a.txt").write_text("alpha as text")
    settings = _caller_hash_settings(tmp_path)
    async with TarnRag(settings) as tarn:
        outcome = await tarn.ingest([str(src)])
        assert [s.document_id for s in outcome.value] == ["a"]  # the first file per stem is kept
        [issue] = outcome.report.issues
        assert issue.severity is Severity.ERROR
        assert issue.subject == str(src / "a.txt")  # the skipped file
        assert "'a'" in issue.message and "a.md" in issue.message  # names the id and the kept file
        assert len((await tarn.docs()).value) == 1  # nothing was overwritten


async def test_ingest_warns_when_replacing_a_document_of_a_different_kind(tmp_path):
    """Cross-call stem reuse: ``a.txt`` ingested earlier, then ``a.md`` from elsewhere — same stem = same
    id, so the replace proceeds (the upsert contract), but the **kind change** is WARNING-reported
    (probably a different document, not a re-ingest). A same-kind re-ingest stays silent."""
    run1, run2 = tmp_path / "one", tmp_path / "two"
    run1.mkdir()
    run2.mkdir()
    (run1 / "a.txt").write_text("alpha as text")
    (run2 / "a.md").write_text("alpha as markdown")
    async with TarnRag(_caller_hash_settings(tmp_path)) as tarn:
        assert (await tarn.ingest([str(run1)])).report.ok  # fresh id — nothing replaced, no issues
        assert (await tarn.ingest([str(run1)])).report.ok  # same-kind re-ingest — the normal upsert, silent

        outcome = await tarn.ingest([str(run2)])  # a.md replaces the txt-sourced document 'a'
        [issue] = outcome.report.issues
        assert issue.severity is Severity.WARNING
        assert issue.subject == str(run2 / "a.md")
        assert "'txt'" in issue.message and "'md'" in issue.message  # names both kinds
        assert [s.status for s in outcome.value] == ["complete"]  # the replace still went through
        assert len((await tarn.docs()).value) == 1  # still one document 'a'


async def test_ingest_stays_silent_over_an_unknown_stored_kind(tmp_path):
    """No spurious warnings where the kind can't be compared: a stored kind of ``'document'`` (the
    pre-fix default — legacy stores) and an incoming file with no extension are both unknown."""
    from tarnrag.contracts import Document

    src = tmp_path / "in"
    src.mkdir()
    (src / "a.md").write_text("alpha as markdown")
    (src / "b").write_text("bare beta")  # no extension — incoming kind unknown
    async with TarnRag(_caller_hash_settings(tmp_path)) as tarn:
        # Seed 'a' the way a pre-fix store looks (source_kind fell back to 'document') and 'b' with a
        # real kind; the a.md replace can't compare against 'document', the 'b' file has no extension.
        await tarn._repository.store_document(Document(content="old alpha", metadata={"source_id": "a"}))
        await tarn._repository.store_document(
            Document(content="old beta", metadata={"source_id": "b", "source_kind": "txt"})
        )
        outcome = await tarn.ingest([str(src)])
        assert outcome.report.ok  # both replacements proceed silently — nothing comparable changed


def test_split_stem_collisions_keeps_the_first_per_stem(tmp_path):
    """The partition itself: first file per stem kept in order; later same-stem files come back as
    ``(skipped, kept)`` pairs — including a same-name file from a different directory."""
    a_md, a_txt = tmp_path / "a.md", tmp_path / "a.txt"
    sub = tmp_path / "sub"
    sub.mkdir()
    a_sub = sub / "a.rst"
    b = tmp_path / "b.md"
    kept, collisions = TarnRag._split_stem_collisions([a_md, a_txt, b, a_sub])
    assert kept == [a_md, b]
    assert collisions == [(a_txt, a_md), (a_sub, a_md)]


async def test_close_releases_built_resources(tmp_path):
    """``close`` releases what the facade BUILT: the generation LLM (when not injected) and the ingestion
    engine's queue — plus the shared store. Recorded through the same ``aclose`` seams the engines use."""
    settings = _plumbing_settings(tmp_path)
    settings.llm.api_key = "k"  # lets the facade construct a backend (lazily connecting — no network)
    tarn = await TarnRag(settings).open()
    await tarn.docs()  # lazily builds the ingestion engine
    tarn._gen()  # lazily builds the generation engine over the built LLM
    closed: list[str] = []

    class _Closable:
        def __init__(self, name: str):
            self._name = name

        async def aclose(self) -> None:
            closed.append(self._name)

    assert tarn._generation.llm is tarn._llm  # generation holds the one LLM the facade built
    tarn._llm = _Closable("llm")
    tarn._ingestion._queue = _Closable("queue")
    await tarn.close()
    assert closed == ["llm", "queue"]  # the built LLM and the engine's queue are both released


async def test_close_leaves_an_injected_llm_to_its_owner(tmp_path):
    """An INJECTED LLM is closed by whoever supplied it, never by the facade."""
    closed: list[str] = []

    class _InjectedLLM(StaticLanguageModel):
        async def aclose(self) -> None:
            closed.append("llm")

    async with TarnRag(_plumbing_settings(tmp_path), llm=_InjectedLLM("{}")) as tarn:
        tarn._gen()
    assert closed == []  # __aexit__ -> close() did not touch the injected model


def test_expand_resolves_files_dirs_and_reports_missing(tmp_path):
    """``_expand`` (path resolution only): a file is taken as-is, a directory contributes the files in it
    (sorted), and a path that matches nothing comes back as missing — never silently dropped."""
    (tmp_path / "a.txt").write_text("a")
    sub = tmp_path / "docs"
    sub.mkdir()
    (sub / "b.txt").write_text("b")
    (sub / "c.txt").write_text("c")
    files, missing = TarnRag._expand([str(tmp_path / "a.txt"), str(sub), "nope.txt"])
    assert [f.name for f in files] == ["a.txt", "b.txt", "c.txt"]  # file, then dir contents
    assert missing == ["nope.txt"]


async def test_gen_builds_the_generation_engine_over_shared_retrieval(tmp_path):
    """``_gen`` lazily builds the generation engine once, over the SHARED retrieval engine + the injected
    LLM (no embedding — the model only runs when you actually ``ask``)."""
    async with TarnRag(_plumbing_settings(tmp_path), llm=StaticLanguageModel("{}")) as tarn:
        gen = tarn._gen()
        assert gen is tarn._gen()                # built once, then cached
        assert gen.retrieval is tarn._retrieval  # wired over the shared retrieval engine
        assert gen.llm is tarn._injected_llm     # the injected model


async def test_ask_without_an_llm_key_fails_fast(tmp_path, monkeypatch):
    """``ask`` needs an LLM: with no injected model and no API key, generation is built lazily and fails
    fast with a clear message — before any retrieval."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    async with TarnRag(_plumbing_settings(tmp_path)) as tarn:  # no llm, no key
        with pytest.raises(RuntimeError, match="needs an LLM key"):
            await tarn.ask("anything")


# ---------------- end-to-end (needs the embedding model) ----------------


def _settings(tmp_path: Path) -> Settings:
    """The sample console config, pointed at a temp store + the fetched model."""
    config = json.loads(Path("examples/console.config.json").read_text())
    config["database"]["document_url"] = f"sqlite:///{tmp_path}/console.db"
    config["embedding"]["model_dir"] = str(MODEL_DIR)
    return Settings(_env_file=None, **config)


def _canned_llm() -> StaticLanguageModel:
    """Routes reasoner replies on the question line; grounding prompts (the fact-checker) say grounded."""

    def _reply(prompt):
        if prompt.system and "fact-checker" in prompt.system:
            return json.dumps({"verdicts": [True, True, True, True]})
        question = prompt.user.split("\n", 1)[0].lower()
        if "pump" in question:
            return json.dumps(
                {"answer": "Check the mechanical seal and bearing lubrication.",
                 "steps": [{"claim": "service the mechanical seal and bearing lubrication", "cited": [1]}]}
            )
        if "capital" in question:
            return json.dumps({"answer": "Paris.", "steps": [{"claim": "the capital of france is paris", "cited": [1]}]})
        return json.dumps({"answer": "I don't know.", "steps": []})

    return StaticLanguageModel(_reply)


@requires_model
async def test_tarnrag_ingest_retrieve_ask_delete(tmp_path):
    pytest.importorskip("onnxruntime")
    pytest.importorskip("tokenizers")
    doc_paths = sorted(str(p) for p in corpus("corpus-1").glob("*.txt"))

    async with TarnRag(_settings(tmp_path), llm=_canned_llm()) as tarn:
        # ingest — the document id is each file's stem; a clean run reports no issues
        ingested = await tarn.ingest(doc_paths)
        assert ingested.report.ok
        statuses = ingested.value
        assert len(statuses) == 3 and all(s.status == "complete" for s in statuses)
        assert {s.document_id for s in statuses} == {"pump-maintenance", "quokka", "tank-inspection"}
        assert len((await tarn.docs()).value) == 3

        # retrieve — the pump question surfaces the pump doc
        hits = (await tarn.retrieve("how should I service a pump")).value
        assert hits and any(h.document_id == "pump-maintenance" for h in hits)

        # ask — an answerable question is grounded; an off-topic one abstains (the cascade + policy)
        answered = (await tarn.ask("How should I service a pump before restarting it?")).value
        assert not answered.abstained and answered.grounded and answered.answer
        off_topic = (await tarn.ask("What is the capital of France?")).value
        assert off_topic.abstained

        # re-ingest one file (same stem -> upsert, still 3 docs) + delete one
        again = await tarn.ingest([doc_paths[0]])
        assert len(again.value) == 1 and len((await tarn.docs()).value) == 3
        assert (await tarn.delete("quokka")).value is True
        assert len((await tarn.docs()).value) == 2


@requires_model
async def test_ingest_reports_a_missing_path_alongside_a_real_one(tmp_path):
    """The mixed case (needs the model to embed the real file): a path that matches nothing is reported,
    while the file that does exist still ingests."""
    pytest.importorskip("onnxruntime")
    pytest.importorskip("tokenizers")
    real = min(str(p) for p in corpus("corpus-1").glob("*.txt"))

    async with TarnRag(_settings(tmp_path), llm=_canned_llm()) as tarn:
        outcome = await tarn.ingest(["does/not/exist.txt", real])
        assert len(outcome.value) == 1 and outcome.value[0].status == "complete"  # the real file ingested
        assert [i.subject for i in outcome.report.issues] == ["does/not/exist.txt"]  # the missing one reported
        assert outcome.report.issues[0].severity is Severity.WARNING


@requires_model
async def test_console_renders_over_the_facade(tmp_path):
    """The rich Console is a thin UI over TarnRag: its handlers run end to end (no engine access of their
    own) and stay alive when a command errors."""
    pytest.importorskip("onnxruntime")
    pytest.importorskip("tokenizers")
    pytest.importorskip("rich")
    from tarnrag.console import Console

    async with TarnRag(_settings(tmp_path), llm=_canned_llm()) as tarn:
        console = Console(tarn)
        # a missing path mixed in -> the report-rendering path runs too
        files = " ".join(str(p) for p in corpus("corpus-1").glob("*.txt"))
        await console._do_ingest(f"does/not/exist.txt {files}")
        await console._do_docs("")
        await console._do_retrieve("how should I service a pump")
        await console._do_ask("How should I service a pump before restarting it?")
        await console._do_ask("What is the capital of France?")  # the abstain branch
        await console._do_delete("quokka")
        await console._do_delete("nope")  # missing id -> handled, not raised
