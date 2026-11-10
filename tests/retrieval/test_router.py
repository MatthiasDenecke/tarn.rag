"""The RoutingRetrievalPipeline: classify a query, then dispatch to the matching sub-pipeline."""

from bausatz import ComponentFactory

from tarnrag.retrieval import (
    GenericQueryClassifier,
    Query,
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
    assert q.query_type == "generic" and len(q.annotations) == 1


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
