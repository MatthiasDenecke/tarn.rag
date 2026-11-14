"""HIGH-07: every engine releases what it built — and only that — so nothing leaks (the retrieval engine's
own LLM client, an API embedder's HTTP client, staged upload copies) and nothing shared is pulled from
under its other users (an injected store, embedder, or LLM)."""

import io

import pytest

from tarnrag.core.engine.config import DatabaseSettings, EmbeddingSettings, Settings
from tarnrag.core.resources.embedder import Embedder
from tarnrag.core.resources.llm import StaticLanguageModel
from tarnrag.ingestion import IngestionEngine
from tarnrag.ingestion.engine.queue import PgQueuerJobQueue
from tarnrag.ingestion.pipeline.embed import EmbedStage
from tarnrag.retrieval import RetrievalEngine
from tarnrag.storage.repository import DocumentRepository


def _settings(tmp_path, **extra) -> Settings:
    return Settings(
        _env_file=None, MODE="embedded", EMBEDDING_DIMENSION=8, embedding=EmbeddingSettings(provider="hash"),
        database=DatabaseSettings(document_url=f"sqlite:///{tmp_path}/own.db"), UPLOAD_DIR=str(tmp_path / "up"),
        **extra,
    )


class _Spy:
    """Records which resources were released, by wrapping their close methods."""

    def __init__(self) -> None:
        self.closed: list[str] = []

    def repo(self, repo: DocumentRepository, name: str = "repo") -> DocumentRepository:
        original = repo.disconnect

        async def disconnect() -> None:
            self.closed.append(name)
            await original()

        repo.disconnect = disconnect
        return repo

    def embedder(self, embedder: Embedder, name: str = "embedder") -> Embedder:
        embedder.close = lambda: self.closed.append(name)
        return embedder

    def llm(self, llm, name: str = "llm"):
        async def aclose() -> None:
            self.closed.append(name)

        llm.aclose = aclose
        return llm


async def _stamped_store(settings: Settings) -> tuple[DocumentRepository, Embedder]:
    repo = await DocumentRepository.create(settings.database, settings.EMBEDDING_DIMENSION)
    embedder = Embedder.create(settings.embedding, settings.EMBEDDING_DIMENSION)
    await IngestionEngine.ensure_index_meta(repo, embedder)
    return repo, embedder


async def test_retrieval_engine_releases_only_what_it_built(tmp_path):
    settings = _settings(tmp_path)
    spy = _Spy()
    repo, embedder = await _stamped_store(settings)
    spy.repo(repo)
    spy.embedder(embedder)

    shared = await RetrievalEngine.create(settings, repository=repo, embedder=embedder)
    spy.llm(shared._llm, "own-llm")  # built from settings: its own
    await shared.aclose()
    assert spy.closed == ["own-llm"]  # the injected store + embedder stay ours

    injected_llm = spy.llm(StaticLanguageModel("{}"), "injected-llm")
    lent = await RetrievalEngine.create(settings, repository=repo, embedder=embedder, llm=injected_llm)
    assert lent._llm is injected_llm  # a composition root shares its one LLM this way
    await lent.aclose()
    assert spy.closed == ["own-llm"]  # nothing of the caller's was closed
    await repo.disconnect()

    spy.closed.clear()
    alone = await RetrievalEngine.create(settings)  # builds its own store + embedder
    spy.repo(alone.repository)
    spy.embedder(alone.embedder)
    spy.llm(alone._llm)
    await alone.aclose()
    assert sorted(spy.closed) == ["embedder", "llm", "repo"]


async def test_ingestion_engine_binds_the_shared_embedder_and_releases_only_its_own(tmp_path):
    settings = _settings(tmp_path)
    spy = _Spy()
    repo, embedder = await _stamped_store(settings)
    spy.repo(repo)
    spy.embedder(embedder)
    engine = await IngestionEngine.create(settings, repository=repo, embedder=embedder)
    [embed] = [s for s in engine.pipeline.stages if isinstance(s, EmbedStage)]
    assert embed._embedder is embedder  # one model, the one whose fingerprint was stamped — not a second copy
    await engine.aclose()
    assert spy.closed == []
    await repo.disconnect()

    spy.closed.clear()
    alone = await IngestionEngine.create(settings)
    spy.repo(alone.repository)
    spy.embedder(alone._owned_embedder)
    await alone.aclose()
    assert sorted(spy.closed) == ["embedder", "repo"]


@pytest.mark.parametrize("failing_step", ["embedder", "queue"])
async def test_a_create_that_fails_part_way_releases_what_it_built(tmp_path, monkeypatch, failing_step):
    """Distributed ``create`` connects the store, then builds the embedder, then connects the queue. When a
    later step fails (no model, the queue DB down), what the earlier steps built is released before the
    error propagates."""
    spy = _Spy()
    real_repo_create, real_embedder_create = DocumentRepository.create, Embedder.create
    sqlite = DatabaseSettings(document_url=f"sqlite:///{tmp_path}/own.db")

    async def repo_create(database, dim):  # a real SQLite store stands in for the Postgres one
        return spy.repo(await real_repo_create(sqlite, dim))

    def embedder_create(embedding, dim):
        if failing_step == "embedder":
            raise FileNotFoundError("no model")
        return spy.embedder(real_embedder_create(embedding, dim))

    async def queue_connect(url, **options):
        raise ConnectionRefusedError("queue DB down")

    monkeypatch.setattr(DocumentRepository, "create", staticmethod(repo_create))
    monkeypatch.setattr(Embedder, "create", staticmethod(embedder_create))
    monkeypatch.setattr(PgQueuerJobQueue, "connect", staticmethod(queue_connect))
    settings = Settings(
        _env_file=None, MODE="distributed", EMBEDDING_DIMENSION=8, embedding=EmbeddingSettings(provider="hash"),
        database=DatabaseSettings(document_url="postgresql://u@localhost/docs", queue_url="postgresql://u@localhost/q"),
    )
    with pytest.raises(FileNotFoundError if failing_step == "embedder" else ConnectionRefusedError):
        await IngestionEngine.create(settings)
    assert sorted(spy.closed) == (["repo"] if failing_step == "embedder" else ["embedder", "repo"])


async def test_embedded_stream_ingest_removes_its_staged_copies(tmp_path):
    settings = _settings(tmp_path, ID_POLICY="caller")
    async with await IngestionEngine.create(settings) as engine:
        ids = await engine.ingest_streams([("notes.txt", io.BytesIO(b"hello stream"))], source_ids=["s"])
        assert (await engine.status(ids[0])).status == "complete"
    staged = list((tmp_path / "up").iterdir()) if (tmp_path / "up").exists() else []
    assert staged == []  # the in-process pipeline is done with them


async def test_a_failed_staging_leaves_no_partial_batch(tmp_path):
    class _Broken(io.RawIOBase):
        def readinto(self, b):
            raise OSError("upload interrupted")

    settings = _settings(tmp_path, ID_POLICY="caller")
    async with await IngestionEngine.create(settings) as engine:
        try:
            await engine.ingest_streams(
                [("a.txt", io.BytesIO(b"fine")), ("b.txt", _Broken())], source_ids=["a", "b"]
            )
            raise AssertionError("expected the broken stream to fail the call")
        except OSError:
            pass
    assert list((tmp_path / "up").iterdir()) == []
