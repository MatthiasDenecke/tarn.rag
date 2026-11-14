"""The console's results / SearchTrace rendering (A3) — pure rendering, no store / embedder / model.

The render helpers (``results_table`` / ``trace_view``) return rich renderables; here we render them to
plain text (a non-terminal ``Console``) and assert the salient content appears. The example runner reuses
these same helpers, so they are part of the console's public surface."""

import io

from rich.console import Console as RichConsole

from tarnrag.console import Console
from tarnrag.contracts import (
    Annotation,
    Candidate,
    ChunkProvenance,
    CorpusStatus,
    RetrievalResult,
)
from tarnrag.generation.types import Citation, GenerationResult, ProofStep
from tarnrag.retrieval.types import Query, RetrieverCandidates, SearchStage, SearchTrace


def _result(chunk_id, text, score, components, document_id):
    return RetrievalResult(
        chunk_id=chunk_id, text=text, score=score, component_scores=components,
        document_id=document_id, source_kind="markdown", standard_id=None, locator=None,
        license_class="public_domain",
    )


def _render(renderable, width=200):
    out = RichConsole(width=width, file=io.StringIO(), force_terminal=False)
    out.print(renderable)
    return out.file.getvalue()


def test_results_table_shows_a_column_per_component_score():
    rows = [
        _result("c1", "storage tank inspection", 0.9, {"dense": 0.8, "sparse": 0.5}, "tank-inspection"),
        _result("c2", "quokka marsupial", 0.4, {"dense": 0.4}, "quokka"),
    ]
    text = _render(Console.create_results_table(rows))
    assert "dense" in text and "sparse" in text          # one column per component present
    assert "tank-inspection" in text and "quokka" in text
    assert "0.800" in text                               # the dense component score, formatted
    assert "Δ" not in text                               # no movement column without prev_order


def test_results_table_movement_column_marks_rank_changes():
    a = _result("a", "alpha", 0.9, {"dense": 0.9}, "doc-a")
    b = _result("b", "bravo", 0.8, {"dense": 0.8}, "doc-b")
    text = _render(Console.create_results_table([a, b], prev_order=["b", "a"]))  # was [b, a], now [a, b]
    assert "Δ" in text
    assert "▲1" in text and "▼1" in text                 # a moved up one, b moved down one


def test_results_table_movement_marks_a_newly_appeared_row():
    parent = _result("parent", "merged section", 0.9, {"dense": 0.9}, "compressor-startup")
    text = _render(Console.create_results_table([parent], prev_order=["leaf1", "leaf2"]))
    assert "＋" in text                                  # the auto-merged parent wasn't a retrieved leaf


def test_trace_view_renders_query_routing_retrievers_and_stages():
    fused = [
        _result("c1", "storage tank inspection", 0.9, {"dense": 0.8, "sparse": 0.5}, "tank-inspection"),
        _result("c2", "quokka marsupial", 0.3, {"dense": 0.3}, "quokka"),
    ]
    trace = SearchTrace(
        query=Query(text="how to inspect a tank", top_k=1),
        per_retriever=[
            RetrieverCandidates("dense", [Candidate("c1", 1, 0.20), Candidate("c2", 2, 0.70)]),
            RetrieverCandidates("sparse", [Candidate("c1", 1, -2.10)]),
        ],
        stages=[SearchStage("fused", fused), SearchStage("final", fused[:1])],
        route_path=[("semantic", "default")],
    )
    text = _render(Console.create_trace_view(trace))
    assert "how to inspect a tank" in text               # the query
    assert "routed" in text and "semantic" in text and "default" in text  # routing decision
    assert "dense" in text and "sparse" in text          # per-retriever sections
    assert "fused" in text and "final" in text           # the per-stage tables
    assert "tank-inspection" in text                     # candidate hydrated from the stage results


def test_trace_view_shows_the_original_wording_of_a_rewritten_follow_up():
    """With conversation memory on, ``explain``'s trace carries the EXECUTED query plus a
    ``contextualized`` annotation holding what was typed — the view shows both, so the rewrite is
    visible exactly where the search is being explained."""
    hits = [_result("c1", "GMV frames require ISO VG 68", 0.9, {"dense": 0.2}, "lubrication-spec")]
    rewritten = Query(
        text="Which oil grade does the GMV frame require?",
        annotations=[
            Annotation(
                producer="condense", type="contextualized",
                value={"original": "Which oil grade does it require?", "turns_used": 1},
                deterministic=False,
            )
        ],
    )
    text = _render(Console.create_trace_view(SearchTrace(query=rewritten, stages=[SearchStage("final", hits)])))
    assert "query: Which oil grade does the GMV frame require?" in text
    assert "asked: Which oil grade does it require?" in text


def test_trace_view_of_a_query_used_as_typed_has_no_asked_line():
    hits = [_result("c1", "alpha", 0.9, {"dense": 0.2}, "doc")]
    text = _render(Console.create_trace_view(SearchTrace(query=Query(text="alpha"), stages=[SearchStage("final", hits)])))
    assert "asked:" not in text


def test_create_answer_view_grounded_shows_answer_and_proof():
    result = GenerationResult(
        answer="Service the mechanical seal first.",
        proof=[
            ProofStep(
                claim="Check the mechanical seal",
                citations=[Citation(chunk_id="c1", document_id="pump-maintenance", header_path=["Centrifugal pump maintenance"])],
                grounded=True,
            ),
            ProofStep(claim="Confirm shaft alignment", citations=[], grounded=False),
        ],
        grounded=True,
    )
    text = _render(Console.create_answer_view(result))
    assert "Service the mechanical seal first." in text
    assert "grounded" in text
    assert "Check the mechanical seal" in text and "Confirm shaft alignment" in text
    assert "✓" in text and "✗" in text                   # per-step grounding marks
    assert "Centrifugal pump maintenance" in text        # the citation's header path


def test_create_answer_view_abstained_shows_only_the_refusal():
    result = GenerationResult(answer="I don't have evidence for that.", abstained=True)
    text = _render(Console.create_answer_view(result))
    assert "abstained" in text
    assert "I don't have evidence for that." in text


def test_create_details_table_surfaces_chunk_id_section_locator_and_license():
    r = RetrievalResult(
        chunk_id="abcdef1234567890", text="t", score=0.9, component_scores={"dense": 0.3},
        document_id="tank-inspection", source_kind="document", standard_id=None,
        locator="§6.4", license_class="public_domain",
        provenance=ChunkProvenance(content_hash="x", header_path=["Tank inspection", "Shell thickness"]),
    )
    text = _render(Console.create_details_table([r]))
    assert "abcdef12" in text                     # the chunk id, shortened to 8 chars
    assert "tank-inspection" in text
    assert "Shell thickness" in text              # the section header path
    assert "§6.4" in text                         # the locator
    assert "public_domain" in text                # the license class


def test_trace_view_includes_overfetch_filter_summary_and_score_legend():
    fused = [_result("c1", "storage tank inspection", 0.9, {"dense": 0.30, "sparse": 0.5}, "tank-inspection")]
    trace = SearchTrace(
        query=Query(text="q", top_k=1),
        per_retriever=[RetrieverCandidates("dense", [Candidate("c1", 1, 0.30)])],
        stages=[SearchStage("fused", fused), SearchStage("final", fused)],
    )
    text = _render(Console.create_trace_view(trace))
    assert "over-fetch" in text and "top-50" in text           # the over-fetch targets
    assert "pre-filter" in text and "EXECUTION" in text and "scope=ALL" in text  # the license/scope filter
    assert "cosine distance" in text                           # the score legend (dense semantics)
    assert "result details" in text and "public_domain" in text  # the details table is appended


def test_create_status_view_summarizes_the_corpus():
    status = CorpusStatus(
        document_count=12, chunk_count=19, embedding_count=19, total_chars=3000,
        min_chars=149, max_chars=400, mean_chars=250.0, median_chars=246.0, mean_chunks_per_doc=1.58,
    )
    text = _render(Console.create_status_view(status))
    assert "documents" in text and "12" in text
    assert "chunks" in text and "19" in text
    assert "median 246" in text and "max 400" in text          # the doc-length distribution line
    assert "3,000" in text                                     # total chars, thousands-separated
