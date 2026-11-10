"""The enricher seam (FR-5): the built-in AcronymEnricher, the EnrichStage driver, and AC-2 — a
custom enricher registers and runs by config, annotating at document- and element-level, no base edits."""

from typing import Literal

import pytest
from bausatz import ComponentFactory
from pydantic import ValidationError

from tarnrag.contracts import (
    Element,
    ElementKind,
    PipelineItem,
    Span,
    StructuredDocument,
)
from tarnrag.ingestion.components.enrichment.acronyms import AcronymEnricher
from tarnrag.ingestion.components.enrichment.enrich import EnrichStage
from tarnrag.ingestion.components.enrichment.enricher import Enricher


def _single(text: str) -> StructuredDocument:
    return StructuredDocument(
        source_id="d1", source_kind="md", extractor="md", text=text, content_hash="h",
        elements=[Element(id="e0", kind=ElementKind.PARAGRAPH, text=text, geometry=[Span(start=0, end=len(text))])],
    )


def test_acronym_enricher_tags_spans():
    doc = _single("Wear PPE near OSHA zones.")
    AcronymEnricher(AcronymEnricher.Config()).enrich(doc)
    anns = doc.elements[0].annotations
    assert {a.value["text"] for a in anns} == {"PPE", "OSHA"}
    assert all(a.producer == "acronyms" and a.type == "entity" and a.deterministic for a in anns)
    ppe = next(a for a in anns if a.value["text"] == "PPE")
    s = ppe.span[0]
    assert doc.text[s.start : s.end] == "PPE"  # the annotation span indexes the document text exactly


def test_enrich_stage_runs_configured_enrichers_in_place():
    stage = EnrichStage(EnrichStage.Config(enrichers=[{"class_name": "acronyms"}]))
    doc = _single("Use PPE.")
    out = next(iter(stage.process(PipelineItem(content="", metadata={}, document=doc))))
    assert out.document is doc  # same document, mutated in place, carried forward
    assert any(a.value["text"] == "PPE" for a in out.document.elements[0].annotations)


def test_enrich_stage_with_no_enrichers_is_a_passthrough():
    doc = _single("Wear PPE.")
    out = next(iter(EnrichStage(EnrichStage.Config()).process(PipelineItem(content="c", metadata={"k": 1}, document=doc))))
    assert out.document is doc and not doc.elements[0].annotations  # nothing added
    assert out.content == "c" and out.metadata == {"k": 1}


def test_enrich_stage_rejects_a_non_enricher_spec():
    # Children are typed (``list[Enricher.Config]``), so a foreign child — a stage, not an Enricher — is
    # rejected when the config is validated, before anything is built or run.
    with pytest.raises(ValidationError, match="enrichers"):
        EnrichStage.Config(enrichers=[{"class_name": "Chunk"}])


# --- AC-2: a custom enricher runs purely by config, at both levels, without editing any base class ---

class _TopicEnricher(Enricher):
    """A user-supplied enricher (registers on definition) — a doc-level topic + an element-level tag."""

    class Config(Enricher.Config):
        class_name: Literal["_test_topic"] = "_test_topic"

    def enrich(self, document: StructuredDocument) -> None:
        self.annotate(document, "topic", {"label": "safety"}, deterministic=False)  # document-level
        for element in document.elements:
            self.annotate(element, "classification", {"length": len(element.text)})  # element-level


def test_custom_enricher_runs_by_config_at_doc_and_element_level():
    stage = ComponentFactory.get().create({"class_name": "Enrich", "enrichers": [{"class_name": "_test_topic"}]})
    doc = _single("Wear gear.")
    out = next(iter(stage.process(PipelineItem(content="", metadata={}, document=doc))))
    assert [a.value["label"] for a in out.document.annotations] == ["safety"]  # document-level
    assert out.document.annotations[0].deterministic is False  # generative findings are flagged (FR-5.3)
    assert out.document.elements[0].annotations[0].type == "classification"  # element-level
