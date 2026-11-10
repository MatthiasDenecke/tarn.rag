"""Loaders for the multi-hop QA benchmarks MOTHRAG published — HotpotQA, 2WikiMultiHopQA, MuSiQue.

Each loader turns a downloaded dataset file into ``BenchItem``s: a ``GenEvalQuery`` (question + gold answer
+ gold supporting sentences) paired with the question's candidate **passages**. This is the *distractor*
setting — every question carries its own ~10 (HotpotQA / 2Wiki) or ~20 (MuSiQue) paragraphs, a few gold +
the rest distractors — which ``benchmark_runner`` ingests per question and runs the generation engine over.

The datasets aren't shipped (sizable, separately licensed); download them and point the loader at the file:

- HotpotQA distractor dev: ``hotpot_dev_distractor_v1.json``  (https://hotpotqa.github.io/)
- 2WikiMultiHopQA dev:     ``dev.json``  (same HotpotQA-style schema)
- MuSiQue (answerable) dev: ``musique_ans_v1.0_dev.jsonl``  (JSON Lines)

Scoring is token-F1 / exact-match (SQuAD-style) over ``GenEvalQuery.answer`` — the metric MOTHRAG reports.
"""

from __future__ import annotations

import itertools
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from tarnrag.eval.generation import GenEvalQuery


def _hf_rows(repo: str, split: str, rows: Iterable[dict] | None, *, config: str | None = None) -> Iterable[dict]:
    """The rows for an HF loader: the injected ``rows`` (tests pass synthetic rows — no network/``datasets``),
    else a streamed ``load_dataset``."""
    if rows is not None:
        return rows
    from datasets import (
        load_dataset,  # pragma: no cover - integration path (needs the datasets pkg + network)
    )

    args = (repo, config) if config else (repo,)  # pragma: no cover
    return load_dataset(*args, split=split, streaming=True)  # pragma: no cover


def _limited(rows: Iterable[dict], limit: int | None) -> Iterable[dict]:
    return itertools.islice(rows, limit) if limit else rows


@dataclass
class BenchItem:
    """One benchmark question: the labeled ``GenEvalQuery`` + the ``(title, text)`` passages to ingest."""

    query: GenEvalQuery
    passages: list[tuple[str, str]]


def corpus_from_items(items: list[BenchItem]) -> list[tuple[str, str]]:
    """The deduped union of all ``(title, text)`` passages across ``items`` — the shared corpus for the
    fullwiki-style setting (retrieve over the whole pool, not a per-question one). Deduped by passage text
    (first title kept); order stable (first occurrence). Build it from the *full* dev set, not the eval
    slice, so the haystack is realistic even when only N questions are scored."""
    seen: set[str] = set()
    corpus: list[tuple[str, str]] = []
    for item in items:
        for title, text in item.passages:
            if text not in seen:
                seen.add(text)
                corpus.append((title, text))
    return corpus


def _yes_no(answer: str) -> bool:
    return answer.strip().lower() in {"yes", "no", "noanswer"}


def _hotpot_item(r: dict) -> BenchItem:
    """One HotpotQA/2Wiki distractor record → ``BenchItem``. The record's ``context`` is
    ``[[title, [sent, …]], …]`` and ``supporting_facts`` is ``[[title, sent_idx], …]`` (the shape both the
    raw JSON and the HF rows are normalized to). Each paragraph becomes a passage; the gold supporting
    *sentences* (resolved by index) drive citation coverage."""
    sents_by_title = {title: sents for title, sents in r["context"]}
    passages = [(title, " ".join(sents)) for title, sents in r["context"]]
    supporting = [
        sents_by_title[title][idx]
        for title, idx in r.get("supporting_facts", [])
        if title in sents_by_title and 0 <= idx < len(sents_by_title[title])
    ]
    answer = r["answer"]
    return BenchItem(
        query=GenEvalQuery(
            text=r["question"],
            answer=answer,
            answer_contains=[] if _yes_no(answer) else [answer],
            supporting=supporting,
        ),
        passages=passages,
    )


def load_hotpotqa(path: str | Path, *, limit: int | None = None) -> list[BenchItem]:
    """HotpotQA / 2WikiMultiHopQA distractor JSON (same schema): a list of
    ``{question, answer, supporting_facts: [[title, sent_idx], …], context: [[title, [sent, …]], …]}``."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return [_hotpot_item(r) for r in raw[: limit or len(raw)]]


# 2WikiMultiHopQA uses the HotpotQA distractor schema.
load_2wiki = load_hotpotqa


def load_hotpotqa_hf(
    split: str = "validation", *, limit: int | None = None, rows: Iterable[dict] | None = None
) -> list[BenchItem]:
    """HotpotQA distractor from HuggingFace (``hotpotqa/hotpot_qa``, config ``distractor``) — the reliable
    source, since the canonical ``curtis.ml.cmu.edu`` host is frequently down. Streams the split (so a
    ``limit`` fetches only what it needs) and normalizes the column-oriented HF rows to the JSON shape
    ``_hotpot_item`` consumes. Needs the ``datasets`` package (the ``benchmarks`` extra)."""
    rows = _hf_rows("hotpotqa/hotpot_qa", split, rows, config="distractor")
    return [_hotpot_item(_hf_hotpot_record(r)) for r in _limited(rows, limit)]


def _hf_hotpot_record(r: dict) -> dict:
    """A HuggingFace ``hotpot_qa`` row (column-oriented ``context``/``supporting_facts``) → the row-oriented
    record shape ``_hotpot_item`` expects."""
    return {
        "question": r["question"],
        "answer": r["answer"],
        "context": list(zip(r["context"]["title"], r["context"]["sentences"])),
        "supporting_facts": list(zip(r["supporting_facts"]["title"], r["supporting_facts"]["sent_id"])),
    }


def _2wiki_hf_record(r: dict) -> dict:
    """A ``voidful/2WikiMultihopQA`` row → the row-oriented record ``_hotpot_item`` expects. That repo's
    encoding is messy: titles are wrapped in literal double-quotes, each paragraph's sentence list is a
    JSON-encoded *string*, and ``sent_id`` is a string — so clean the quotes, ``json.loads`` the sentences,
    and ``int`` the index."""

    def title(t: str) -> str:
        return t.strip().strip('"')

    def sents(s: object) -> list[str]:
        return json.loads(s) if isinstance(s, str) else list(s)

    return {
        "question": r["question"],
        "answer": r["answer"],
        "context": [(title(t), sents(s)) for t, s in r["context"]],
        "supporting_facts": [(title(t), int(i)) for t, i in r["supporting_facts"]],
    }


def _musique_item(r: dict) -> BenchItem:
    """One MuSiQue record → ``BenchItem``. ``paragraphs`` carry ``is_supporting``; ``answerable=False`` maps
    to ``should_abstain``; ``answer_aliases`` (acceptable alternative golds) ride along for max-over scoring."""
    paragraphs = r.get("paragraphs", [])
    passages = [(p.get("title", ""), p.get("paragraph_text", "")) for p in paragraphs]
    supporting = [p.get("paragraph_text", "") for p in paragraphs if p.get("is_supporting")]
    answerable = r.get("answerable", True)
    answer = r.get("answer", "") or ""
    aliases = list(r.get("answer_aliases", []) or [])
    return BenchItem(
        query=GenEvalQuery(
            text=r["question"],
            answer=answer,
            answer_aliases=aliases,
            answer_contains=[] if _yes_no(answer) else ([answer, *aliases] if answer else []),
            should_abstain=not answerable,
            supporting=supporting,
        ),
        passages=passages,
    )


def load_musique(path: str | Path, *, limit: int | None = None) -> list[BenchItem]:
    """MuSiQue JSON Lines: one object per line —
    ``{question, answer, answer_aliases, answerable, paragraphs: [{title, paragraph_text, is_supporting}, …]}``."""
    lines = [ln for ln in Path(path).read_text(encoding="utf-8").splitlines() if ln.strip()]
    return [_musique_item(json.loads(ln)) for ln in lines[: limit or len(lines)]]


def load_2wiki_hf(
    split: str = "validation", *, limit: int | None = None, rows: Iterable[dict] | None = None
) -> list[BenchItem]:
    """2WikiMultiHopQA distractor from HuggingFace (``voidful/2WikiMultihopQA``). Streams the split and
    normalizes the messy encoding (see ``_2wiki_hf_record``). Needs the ``datasets`` package."""
    rows = _hf_rows("voidful/2WikiMultihopQA", split, rows)
    return [_hotpot_item(_2wiki_hf_record(r)) for r in _limited(rows, limit)]


def load_musique_hf(
    split: str = "validation", *, limit: int | None = None, rows: Iterable[dict] | None = None
) -> list[BenchItem]:
    """MuSiQue from HuggingFace (``dgslibisey/MuSiQue``). Streams the split, keeps only **answerable**
    questions (MOTHRAG reports F1/EM on the answerable benchmark), and builds the same items as the file
    loader. Needs the ``datasets`` package."""
    rows = _hf_rows("dgslibisey/MuSiQue", split, rows)
    answerable = (r for r in rows if r.get("answerable", True))
    return [_musique_item(r) for r in _limited(answerable, limit)]


# dataset name -> file loader (takes a downloaded path), for the CLI.
LOADERS = {"hotpotqa": load_hotpotqa, "2wiki": load_2wiki, "musique": load_musique}

# dataset name -> HuggingFace loader (no local file needed), for `--hf`.
HF_LOADERS = {"hotpotqa": load_hotpotqa_hf, "2wiki": load_2wiki_hf, "musique": load_musique_hf}
