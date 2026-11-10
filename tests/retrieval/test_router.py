"""The RoutingRetrievalPipeline: classify a query, then dispatch to the matching sub-pipeline."""

from typing import Literal

from bausatz import ComponentFactory

from tarnrag.retrieval import (
    GenericQueryClassifier,
    Query,
    QueryClassifier,
    RetrievalPipeline,
    RoutingRetrievalPipeline,
    Searcher,
    StructuralQueryClassifier,
)


class _RecordingSearcher(Searcher):
    """A fake route: records the query it received and returns its marker (no store needed). Not
    registered (its Config pins no ``class_name``) — it's constructed directly and injected."""

    class Config(Searcher.Config):
        pass

    def __init__(self, marker: str) -> None:
        super().__init__(self.Config())
        self.marker = marker
        self.seen: Query | None = None

    async def search(self, query, ctx, trace=None):
        self.seen = query
        return [self.marker]


def _router(classifier, routes, default) -> RoutingRetrievalPipeline:
    """A router with its children injected — bypasses the factory to unit-test the dispatch logic."""
    r = RoutingRetrievalPipeline(RoutingRetrievalPipeline.Config())
    r._classifier, r._routes, r._default = classifier, routes, default
    r._children_built = True
    return r


async def test_classifies_then_dispatches_to_the_matching_route():
    lex, sem, dflt = _RecordingSearcher("LEX"), _RecordingSearcher("SEM"), _RecordingSearcher("DEF")
    router = _router(
        StructuralQueryClassifier(StructuralQueryClassifier.Config()),
        {"lexical": lex, "semantic": sem},
        dflt,
    )
    assert await router.search(Query(text="shell thickness ultrasonic testing"), None) == ["LEX"]
    assert lex.seen.query_type == "lexical"  # the route saw the now-classified query
    assert await router.search(Query(text="How do I prevent rust?"), None) == ["SEM"]


async def test_supplied_query_type_wins_and_skips_the_classifier():
    lex, sem, dflt = _RecordingSearcher("LEX"), _RecordingSearcher("SEM"), _RecordingSearcher("DEF")
    router = _router(
        StructuralQueryClassifier(StructuralQueryClassifier.Config()),
        {"lexical": lex, "semantic": sem},
        dflt,
    )
    # Clearly-semantic text, but the caller pinned query_type -> the classifier must not run.
    out = await router.search(Query(text="How do I prevent rust?", query_type="lexical"), None)
    assert out == ["LEX"] and lex.seen.annotations == []  # no annotation => classifier was skipped


async def test_unknown_type_falls_through_to_default():
    dflt = _RecordingSearcher("DEF")
    # Generic tags "generic" -> not in routes -> the default route (and it annotated the query).
    router = _router(
        GenericQueryClassifier(GenericQueryClassifier.Config()), {"lexical": _RecordingSearcher("LEX")}, dflt
    )
    q = Query(text="whatever this is")
    assert await router.search(q, None) == ["DEF"]
    assert dflt.seen.query_type == "generic" and len(dflt.seen.annotations) == 1  # the route saw it
    assert q.query_type == "" and q.annotations == []  # classified on a copy: the caller's query is untouched


class _LabelClassifier(QueryClassifier):
    """A test classifier stamping one fixed label (and an annotation) — stands in for a second taxonomy."""

    class Config(QueryClassifier.Config):
        class_name: Literal["_test_label"] = "_test_label"

    def __init__(self, label: str) -> None:
        super().__init__(self.Config())
        self.label = label

    def classify(self, query, ctx) -> None:
        query.query_type = self.label
        self._annotate(query, "query_type", {"label": self.label})


async def test_a_nested_router_runs_its_own_classifier():
    """MED-02: the outer router's label is ITS route key — a nested router must classify by its own
    taxonomy instead of dispatching on the outer label (which its route table doesn't know)."""
    leaf, inner_default = _RecordingSearcher("INNER-LEAF"), _RecordingSearcher("INNER-DEFAULT")
    inner = _router(_LabelClassifier("numeric"), {"numeric": leaf}, inner_default)
    outer = _router(_LabelClassifier("table"), {"table": inner}, _RecordingSearcher("OUTER-DEFAULT"))
    assert await outer.search(Query(text="total revenue 2019"), None) == ["INNER-LEAF"]
    assert leaf.seen.query_type == "numeric"  # the inner classification, not the outer "table"
    assert [a.value["label"] for a in leaf.seen.annotations] == ["table", "numeric"]  # both recorded


async def test_a_supplied_query_type_passes_through_nested_routers():
    """A caller-pinned query_type is the caller's routing decision — honored at every level."""
    leaf = _RecordingSearcher("INNER-LEAF")
    inner = _router(_LabelClassifier("numeric"), {"table": leaf}, _RecordingSearcher("INNER-DEFAULT"))
    outer = _router(_LabelClassifier("other"), {"table": inner}, _RecordingSearcher("OUTER-DEFAULT"))
    assert await outer.search(Query(text="x", query_type="table"), None) == ["INNER-LEAF"]
    assert leaf.seen.annotations == []  # neither classifier ran


def test_factory_builds_a_router_as_a_searcher():
    spec = {
        "class_name": "routing_retrieval_pipeline",
        "classifier": {"class_name": "structural"},
        "routes": {
            "lexical": {
                "class_name": "retrieval_pipeline",
                "retrievers": [{"class_name": "sparse"}],
                "fuser": {"class_name": "identity"},
            },
            "semantic": {"class_name": "retrieval_pipeline", "retrievers": [{"class_name": "dense"}]},
        },
    }
    router = ComponentFactory.get().create_as(spec, Searcher)
    assert isinstance(router, RoutingRetrievalPipeline)


def test_default_router_is_just_the_default_pipeline():
    # All defaults (generic classifier, no routes, dense default): builds, and behaves as a single pipeline.
    router = ComponentFactory.get().create_as({"class_name": "routing_retrieval_pipeline"}, Searcher)
    router.ensure_built()
    assert router._routes == {} and isinstance(router._default, RetrievalPipeline)
