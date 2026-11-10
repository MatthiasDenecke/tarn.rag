"""
SQLite adapter (SQLAlchemy Core + aiosqlite) — also the §8 retrieval store.

The normal §8 tables (documents / chunks / method_chunks / index_meta) are SQLAlchemy ``Table``s
defined in the base; the dense + sparse retrieval indexes are SQLite-extension **virtual tables**:
``vec_chunks`` (sqlite-vec, dense KNN) and ``fts_chunks`` (FTS5, sparse).

Those virtual-table operations are intentionally **raw SQL** (``exec_driver_sql``), not SQLAlchemy
Core, for two reasons:

* SQLAlchemy doesn't model virtual tables — no ``Table`` emits ``CREATE VIRTUAL TABLE ... USING
  ...`` (true of built-in FTS5, not just the proprietary sqlite-vec).
* The KNN query uses sqlite-vec's proprietary ``MATCH`` operator + synthetic ``distance`` column,
  which has no SQLAlchemy expression.

A few statements that touch only the normal tables (the over-fetch joins, the ``chunk_id``
subqueries) are kept raw too, deliberately, so the whole sqlite-vec/FTS layer reads as one block.
``hydrate`` reads only normal tables, so it is shared Core in the base (``ReadAssembler.hydrate``).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import sqlite_vec
from sqlalchemy import Table, Text, event
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.ext.asyncio import AsyncConnection

from tarnrag.contracts import (
    Candidate,
    Chunk,
    ChunkFilter,
    Embedding,
    is_retrieval_unit,
)
from tarnrag.core.text import lexical_query
from tarnrag.storage.repository.base import DocumentRepository

_WORD = re.compile(r"\w+")  # FTS5 unicode61 splits a phrase into these word pieces


class SqliteRepository(DocumentRepository):
    """
    SQLite adapter and the §8 retrieval store: dense vectors in ``vec_chunks`` (sqlite-vec) and
    sparse text in ``fts_chunks`` (FTS5). See the module docstring for why that layer is raw SQL.
    """

    def __init__(self, connection_url: str, embedding_dimension: int = 384):
        super().__init__(connection_url, embedding_dimension)

        @event.listens_for(self.engine.sync_engine, "connect")
        def _on_connect(dbapi_connection, _record):
            # SQLite enforces foreign keys / ON DELETE CASCADE only when
            # ``PRAGMA foreign_keys=ON``, which is per-connection.
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()
            # Load sqlite-vec so vec0 KNN is available on every connection (see
            # ``_load_sqlite_vec`` for the ``._conn`` rationale). This is the foundation for
            # folding the §8 retrieval index into this repository.
            self._load_sqlite_vec(dbapi_connection)

    @staticmethod
    def _load_sqlite_vec(dbapi_connection) -> None:
        """
        Load the sqlite-vec extension onto a freshly-opened connection (called from the
        per-connection ``connect`` event), so vec0 dense-KNN is available on every connection.

        Why this reaches a **private aiosqlite attribute** (``._conn``)
        --------------------------------------------------------------
        Loading a SQLite loadable extension needs the C-API calls
        ``conn.enable_load_extension(True)`` / ``conn.load_extension(path)`` on the **real,
        synchronous** ``sqlite3.Connection``. Neither layer above us offers that synchronously:

        * ``dbapi_connection`` here is SQLAlchemy's async adapter
          (``AsyncAdapt_aiosqlite_connection``) — it does **not** expose ``enable_load_extension``.
        * ``dbapi_connection.driver_connection`` is the ``aiosqlite.Connection`` (public SQLAlchemy
          API), but aiosqlite's ``enable_load_extension`` / ``load_extension`` are **async**
          coroutines — unusable from this *synchronous* ``connect`` event.

        So we descend one more level, to the ``sqlite3.Connection`` that aiosqlite wraps, and load
        on it directly. aiosqlite opens that connection with ``check_same_thread=False``, so calling
        its sync methods from this hook is safe::

            dbapi_connection                  # AsyncAdapt_aiosqlite_connection  (SQLAlchemy adapter)
                .driver_connection            # aiosqlite.Connection             (public SQLAlchemy API)
                ._conn                        # sqlite3.Connection               (PRIVATE aiosqlite attr) <-- here

        ``._conn`` is a **private aiosqlite attribute** and is therefore the single fragile point
        of this integration: an aiosqlite-internal rename would silently break extension loading.
        It is guarded by ``tests/storage/test_sqlite_vec_under_aiosqlite.py``, which fails loudly
        if the path changes. If aiosqlite ever exposes a public *sync* accessor (or SQLAlchemy
        gains an async ``connect`` hook), switch to it and delete this note.
        """
        raw_sqlite3_connection = dbapi_connection.driver_connection._conn  # PRIVATE aiosqlite attr
        raw_sqlite3_connection.enable_load_extension(True)
        sqlite_vec.load(raw_sqlite3_connection)
        raw_sqlite3_connection.enable_load_extension(False)

    async def connect(self) -> None:
        # SQLite won't create the parent directory — ensure it exists (no-op for in-memory).
        db_file = self.engine.url.database
        if db_file and db_file != ":memory:":
            Path(db_file).parent.mkdir(parents=True, exist_ok=True)
        await super().connect()

    async def _after_create_schema(self, conn: AsyncConnection) -> None:
        """
        Build the §8 sqlite-vec dense index + FTS5 sparse index (virtual tables; the sqlite-vec
        extension is loaded per connection by the connect hook above).
        """
        await conn.exec_driver_sql(
            "CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0("
            f"chunk_id TEXT PRIMARY KEY, embedding float[{self.embedding_dimension}])"
        )
        await conn.exec_driver_sql(
            "CREATE VIRTUAL TABLE IF NOT EXISTS fts_chunks USING fts5("
            "chunk_id UNINDEXED, text, tokenize='unicode61')"
        )

    def _driver_url(self, url: str) -> str:
        path = url.replace("sqlite:///", "").replace("sqlite://", "")
        return f"sqlite+aiosqlite:///{path}"

    def _vector_type(self):
        return Text

    def _encode_vector(self, vector: list[float]):
        return json.dumps(list(vector))

    def _decode_vector(self, stored) -> list[float]:
        return json.loads(stored)

    def _dialect_insert(self, table: Table):
        return sqlite_insert(table)

    async def store_embeddings(self, embeddings: list[Embedding]) -> list[str]:
        """
        §8: dense vectors live in the sqlite-vec ``vec_chunks`` virtual table, not the embeddings
        table (which stays empty on SQLite). Stores exactly the DTO — ``chunk_id`` + ``vector`` —
        idempotently (``INSERT OR REPLACE`` on the chunk_id key).
        """
        if not embeddings:
            return []
        rows = [(e.chunk_id, sqlite_vec.serialize_float32(e.vector)) for e in embeddings]
        async with self.engine.begin() as conn:
            await conn.exec_driver_sql(
                "INSERT OR REPLACE INTO vec_chunks(chunk_id, embedding) VALUES (?, ?)", rows
            )
        return [e.chunk_id for e in embeddings]

    async def dense_knn(
        self, query_vec: list[float], k: int, filter: ChunkFilter | None = None
    ) -> list[Candidate]:
        """
        Exact KNN over ``vec_chunks`` (sqlite-vec), nearest first. Raw SQL: ``MATCH`` and the
        synthetic ``distance`` column are sqlite-vec's proprietary query API (see module docstring).
        The result is the ``k`` nearest *retrieval units* (leaf chunks, ``level = 0``) that pass
        ``filter``: sqlite-vec picks its k nearest before a join can filter, so we over-fetch a window,
        join ``chunks`` to drop the rest, and backfill (``_overfetch``, ModusQ §5.4). The leaf predicate
        also holds for an index built before section parents stopped being embedded.
        """
        q = sqlite_vec.serialize_float32(query_vec)
        where, params = self._search_where(filter, "c")
        async with self.engine.connect() as conn:
            total = (await conn.exec_driver_sql("SELECT count(*) FROM vec_chunks")).fetchone()[0]

            async def page(window: int) -> list[tuple[str, float]]:
                return (
                    await conn.exec_driver_sql(
                        "SELECT v.chunk_id, v.distance FROM "
                        "(SELECT chunk_id, distance FROM vec_chunks WHERE embedding MATCH ? "
                        "ORDER BY distance LIMIT ?) v JOIN chunks c ON c.chunk_id = v.chunk_id "
                        f"WHERE {where} ORDER BY v.distance",
                        (q, window, *params),
                    )
                ).fetchall()

            return await self._overfetch(k, total, page)

    async def sparse_search(
        self, query_text: str, k: int, filter: ChunkFilter | None = None
    ) -> list[Candidate]:
        """§8 sparse retrieval over FTS5 ``fts_chunks`` (BM25, best first). Raw SQL: ``MATCH`` + the
        ``bm25()`` rank are FTS5's query API. ``raw_score`` is the FTS5 BM25 value (lower = better).
        Restricted to retrieval units and ``filter`` as in :meth:`dense_knn` (over-fetch the top window,
        join ``chunks``, keep the permitted leaves, backfill)."""
        match = self._fts_query(query_text)
        if not match:
            return []
        where, params = self._search_where(filter, "c")
        async with self.engine.connect() as conn:
            total = (await conn.exec_driver_sql("SELECT count(*) FROM fts_chunks")).fetchone()[0]

            async def page(window: int) -> list[tuple[str, float]]:
                return (
                    await conn.exec_driver_sql(
                        "SELECT f.chunk_id, f.score FROM "
                        "(SELECT chunk_id, bm25(fts_chunks) AS score FROM fts_chunks "
                        "WHERE fts_chunks MATCH ? ORDER BY score LIMIT ?) f "
                        "JOIN chunks c ON c.chunk_id = f.chunk_id "
                        f"WHERE {where} ORDER BY f.score",
                        (match, window, *params),
                    )
                ).fetchall()

            return await self._overfetch(k, total, page)

    @classmethod
    def _search_where(cls, filter: ChunkFilter | None, alias: str) -> tuple[str, list]:
        """The full search predicate over the joined ``chunks`` row: retrieval units only (a section
        parent is hydrated, never searched) AND, when given, the permitted-chunk ``filter``."""
        where, params = cls._chunk_filter_sql(filter, alias) if filter is not None else ("1", [])
        return f"{alias}.level = 0 AND ({where})", params

    @staticmethod
    def _chunk_filter_sql(filter: ChunkFilter, alias: str) -> tuple[str, list]:
        """Build the permitted-chunk ``WHERE`` fragment + params for ``filter`` over the ``chunks`` row
        aliased ``alias`` (used in the over-fetch join). Returns ``("1", [])`` when nothing is restricted;
        an empty ``method_scope`` yields ``"0"`` (nothing permitted). The SQLite (raw-SQL) counterpart of
        ``PostgresRepository._chunk_filter_condition``."""
        clauses: list[str] = []
        params: list = []
        if filter.require_available:
            clauses.append(f"{alias}.available = 1")
        if filter.require_grounding:
            clauses.append(f"{alias}.ai_grounding_allowed = 1")
        if filter.license_classes is not None:
            if not filter.license_classes:
                clauses.append("0")  # empty permitted set -> nothing permitted
            else:
                placeholders = ", ".join("?" * len(filter.license_classes))
                clauses.append(f"{alias}.license_class IN ({placeholders})")
                params.extend(filter.license_classes)
        if filter.method_scope is not None:
            if not filter.method_scope:
                clauses.append("0")  # empty scope -> nothing permitted
            else:
                ors: list[str] = []
                for ref in filter.method_scope:
                    if ref.method_version is None:
                        ors.append("method_id = ?")
                        params.append(ref.method_id)
                    else:
                        ors.append("(method_id = ? AND method_version = ?)")
                        params.extend([ref.method_id, ref.method_version])
                clauses.append(
                    f"{alias}.chunk_id IN (SELECT chunk_id FROM method_chunks WHERE {' OR '.join(ors)})"
                )
        return (" AND ".join(clauses) if clauses else "1", params)

    @staticmethod
    def _fts_query(text: str) -> str:
        """A safe FTS5 ``MATCH`` expression honoring exact-match intent — the FTS5 rendering of the shared
        :func:`~tarnrag.core.text.lexical_query` parse (Postgres renders the same parse as a tsquery):

        - a double-quoted span is a REQUIRED phrase (tokens adjacent, in order);
        - an identifier whose punctuation splits under unicode61 (``6.4.2``, ``2024-01``, ``§6.4``) is a
          required phrase too — OR-ing its pieces would match them scattered anywhere in a chunk;
        - everything else stays an OR of quoted tokens (recall-first).

        Required parts are ``AND``-ed with the OR-group (explicit — FTS5's implicit AND only joins bare
        phrases, not a phrase with a parenthesized group); all tokens are quoted so query punctuation
        can't trip the FTS5 parser. Empty ⇒ no match (caller returns nothing)."""
        parsed = lexical_query(text)
        parts = [f'"{" ".join(_WORD.findall(phrase.lower()))}"' for phrase in parsed.phrases]
        if parsed.terms:
            group = " OR ".join(f'"{t}"' for t in parsed.terms)
            parts.append(f"({group})" if parts else group)
        return " AND ".join(parts)

    # ----- §8 search-index hooks (vec0 + FTS5 live outside the FK graph) -----

    async def _index_chunk_text(
        self, conn: AsyncConnection, ids: list[str], chunks: list[Chunk]
    ) -> None:
        # Only retrieval units go into the sparse index — a section parent is hydrated, never searched
        # (indexing its text again would also skew BM25's document frequencies).
        rows = [(cid, ch.content) for cid, ch in zip(ids, chunks) if is_retrieval_unit(ch.provenance)]
        if rows:
            await conn.exec_driver_sql("INSERT INTO fts_chunks(chunk_id, text) VALUES (?, ?)", rows)

    async def _clear_chunk_index(self, conn: AsyncConnection, document_id: str) -> None:
        # A subquery, not an expanded id list: one bind parameter however many chunks the document has
        # (an IN list of every chunk id hits SQLITE_MAX_VARIABLE_NUMBER on a large enough document).
        for table in ("vec_chunks", "fts_chunks"):
            await conn.exec_driver_sql(
                f"DELETE FROM {table} WHERE chunk_id IN "
                "(SELECT chunk_id FROM chunks WHERE document_id = ?)",
                (document_id,),
            )

    async def _count_doc_embeddings(self, conn: AsyncConnection, document_id: str) -> int:
        row = (
            await conn.exec_driver_sql(
                "SELECT count(*) FROM vec_chunks WHERE chunk_id IN "
                "(SELECT chunk_id FROM chunks WHERE document_id = ?)",
                (document_id,),
            )
        ).fetchone()
        return row[0]

    async def _embedding_counts_by_document(self, conn: AsyncConnection) -> dict[str, int]:
        rows = (
            await conn.exec_driver_sql(
                "SELECT c.document_id, count(v.chunk_id) FROM chunks c "
                "JOIN vec_chunks v ON v.chunk_id = c.chunk_id GROUP BY c.document_id"
            )
        ).fetchall()
        return {doc_id: cnt for doc_id, cnt in rows}
