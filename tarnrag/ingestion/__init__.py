"""Ingestion domain — public surface.

``IngestionEngine`` is the high-level producer/query facade (``await IngestionEngine.create()``);
``run_worker`` is the consumer entry point for distributed mode. The result types describe what
the engine's calls return. Everything else (orchestrator, queue, worker, stages) is internal.
"""

# Importing each capability package / stage module self-registers its stages + components with the
# global ComponentFactory, so Pipeline.from_spec can build them from Settings.components.
from tarnrag.ingestion.components import chunking, enrichment, extraction  # noqa: F401
from tarnrag.ingestion.engine.engine import IngestionEngine, run_worker
from tarnrag.ingestion.engine.types import DocumentStatus, DocumentSummary
from tarnrag.ingestion.pipeline import clean_normalize, embed  # noqa: F401

__all__ = ["DocumentStatus", "DocumentSummary", "IngestionEngine", "run_worker"]
