"""LoadAndParseStage: config-driven extractor routing -> a StructuredDocument on item.document."""

from tarnrag.contracts import ElementKind, PipelineItem
from tarnrag.ingestion.components.extraction.load_parse import LoadAndParseStage
from tarnrag.ingestion.pipeline.clean_normalize import CleanAndNormalizeStage
from tarnrag.ingestion.pipeline.pipeline import Pipeline


def _run(item: PipelineItem, **config) -> PipelineItem:
    stage = LoadAndParseStage(LoadAndParseStage.Config(**config))
    return next(iter(stage.process(item)))


def test_routes_by_extension_to_the_right_extractor(tmp_path):
    p = tmp_path / "doc.md"
    p.write_text("# Title\n\nBody.", encoding="utf-8")
    out = _run(PipelineItem(content="", metadata={"source_path": str(p), "source_id": "d1"}))
    assert out.document is not None and out.document.extractor == "markdown"
    assert out.content == out.document.text  # content is the normalized text view
    assert any(e.kind == ElementKind.HEADING and e.text == "Title" for e in out.document.elements)


def test_content_only_falls_back_to_plain_text():
    out = _run(PipelineItem(content="hello world", metadata={"source_id": "d1"}))
    assert out.document.extractor == "plain_text"
    assert out.content == "hello world"


def test_metadata_extractor_override_wins(tmp_path):
    p = tmp_path / "doc.md"
    p.write_text("# H\n\nx", encoding="utf-8")
    out = _run(PipelineItem(content="", metadata={"source_path": str(p), "extractor": "plain_text"}))
    assert out.document.extractor == "plain_text"  # the override beats the .md route


def test_route_is_config_driven(tmp_path):
    # point the pdf route at a different extractor via config (plain_text here, to avoid pdf deps)
    p = tmp_path / "doc.pdf"
    p.write_text("raw text", encoding="utf-8")
    out = _run(
        PipelineItem(content="", metadata={"source_path": str(p)}),
        routes={"pdf": {"class_name": "plain_text"}},
    )
    assert out.document.extractor == "plain_text"


def test_unknown_kind_falls_back_to_default_extractor():
    out = _run(PipelineItem(content="x", metadata={"source_type": "rtf"}))
    assert out.document.extractor == "plain_text"


def test_extractors_are_long_lived_built_once():
    stage = LoadAndParseStage(LoadAndParseStage.Config())
    list(stage.process(PipelineItem(content="a", metadata={"source_id": "1"})))
    plain = stage._default
    list(stage.process(PipelineItem(content="b", metadata={"source_id": "2"})))
    assert stage._default is plain  # built once in _build_children, reused (not per item)


def test_extractors_build_through_the_factory_when_built_via_components():
    # via the framework (ComponentFactory.create -> _build_children): the container's children are
    # already built, so process needs no on-the-fly instantiation for a routed kind.
    from bausatz import ComponentFactory

    stage = ComponentFactory.get().create({"class_name": "LoadAndParse"})
    assert isinstance(stage, LoadAndParseStage)
    assert stage._default is not None  # _build_children ran during the factory build
    out = next(iter(stage.process(PipelineItem(content="hi", metadata={"source_id": "d1"}))))
    assert out.document.extractor == "plain_text"


def test_document_survives_a_mapper_stage():
    pipe = Pipeline.from_stages([
        LoadAndParseStage(LoadAndParseStage.Config()),
        CleanAndNormalizeStage(CleanAndNormalizeStage.Config()),
    ])
    [out] = list(pipe.run([PipelineItem(content="hello", metadata={"source_id": "d1"})]))
    assert out.document is not None and out.document.extractor == "plain_text"  # survived Clean
