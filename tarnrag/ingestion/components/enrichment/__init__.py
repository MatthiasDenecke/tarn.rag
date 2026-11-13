"""Enrichment: annotate the loaded ``StructuredDocument`` with typed annotations (NER, topic, …).

Importing this package registers the ``Enrich`` stage and the built-in enrichers (``acronyms``) with
the global ``ComponentFactory``, so ``Pipeline.from_spec`` / ``ComponentFactory`` can build them. Users
register their own ``Enricher`` components on the same framework. The enrichers are imported before
``EnrichStage`` (the driver builds enricher children).
"""

from tarnrag.ingestion.components.enrichment.acronyms import AcronymEnricher
from tarnrag.ingestion.components.enrichment.enrich import EnrichStage
from tarnrag.ingestion.components.enrichment.enricher import Enricher

__all__ = ["AcronymEnricher", "EnrichStage", "Enricher"]
