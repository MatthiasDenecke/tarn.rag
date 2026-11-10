"""tarnrag — RAG ingestion + retrieval + generation engines.

Public API::

    from tarnrag import IngestionEngine, RetrievalEngine, run_worker, Query, DocumentStatus
    from tarnrag import TarnRag  # the high-level facade over all three engines
    from tarnrag import Outcome, Report, Issue, Severity  # what every facade call returns
"""

from tarnrag.ingestion import (
    DocumentStatus,
    DocumentSummary,
    IngestionEngine,
    run_worker,
)
from tarnrag.report import Issue, Outcome, Report, Severity
from tarnrag.retrieval import (
    MethodRef,
    Query,
    RetrievalEngine,
    RetrievalError,
    RetrievalResult,
    SearchTrace,
)
from tarnrag.tarnrag import TarnRag  # imported last — wires the three engines together

__all__ = [
    "DocumentStatus",
    "DocumentSummary",
    "IngestionEngine",
    "Issue",
    "MethodRef",
    "Outcome",
    "Query",
    "Report",
    "RetrievalEngine",
    "RetrievalError",
    "RetrievalResult",
    "SearchTrace",
    "Severity",
    "TarnRag",
    "run_worker",
]
