"""HIGH-02: a re-ingest starts the document's job history afresh — a stale ``failed`` row from an earlier,
superseded run must not pin the document to ``failed`` forever."""

from typing import Literal

from tarnrag.core.engine.config import DatabaseSettings, EmbeddingSettings, Settings
from tarnrag.ingestion import IngestionEngine
from tarnrag.ingestion.components.extraction.extractor import Extractor, Source


class _BrokenExtractor(Extractor):
    """An extractor that always fails — stands in for a corrupt upload."""

    class Config(Extractor.Config):
        class_name: Literal["_broken_extractor"] = "_broken_extractor"

    def extract(self, source: Source):
        raise ValueError("corrupt source")


async def test_a_failed_document_reingested_successfully_reports_complete(tmp_path):
    settings = Settings(
        _env_file=None, MODE="embedded", ID_POLICY="caller", EMBEDDING_DIMENSION=16,
        embedding=EmbeddingSettings(provider="hash"),
        database=DatabaseSettings(document_url=f"sqlite:///{tmp_path}/status.db"),
    )
    async with await IngestionEngine.create(settings) as engine:
        await engine.ingest_content([{"source_id": "d", "content": "text"}], extractor="_broken_extractor")
        assert (await engine.status("d")).status == "failed"  # the first run dead-lettered

        await engine.ingest_content([{"source_id": "d", "content": "the fixed text"}])
        status = await engine.status("d")
        assert status.status == "complete"  # the stale failed row no longer wins
        assert status.chunk_count == status.embedding_count == 1
