"""PostgreSQL adapter (SQLAlchemy Core + asyncpg + pgvector).

§8 dense retrieval (``dense_knn``) via the pgvector ``<=>`` cosine-distance operator. Requires the
``postgres`` extra (``asyncpg``, ``pgvector``); not imported on the SQLite path.
"""

from __future__ import annotations

from sqlalchemy import Table, and_, false, func, or_, select, true
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncConnection
from sqlalchemy.sql.elements import ColumnElement

from tarnrag.contracts import Candidate, ChunkFilter, ChunkRecord
from tarnrag.storage.repository.base import DocumentRepository


class PostgresRepository(DocumentRepository):
    """
    PostgreSQL adapter. Inherits the portable upsert + CRUD from the base; supplies
    the asyncpg driver URL, the pgvector column, and pgvector cosine search.
    """

    def _driver_url(self, url: str) -> str:
        return url.replace("postgresql://", "postgresql+asyncpg://", 1)

    def _vector_type(self):
        return Vector(self.embedding_dimension)

    def _encode_vector(self, vector: list[float]):
        return list(vector)  # pgvector accepts a Python list directly

    def _decode_vector(self, stored) -> list[float]:
        return list(stored)

    def _dialect_insert(self, table: Table):
        return pg_insert(table)

    _IVFFLAT_LISTS = 100  # ivfflat partition count; the filtered path sets probes = this for full recall

    async def _before_create_schema(self, conn: AsyncConnection) -> None:
        await conn.exec_driver_sql("CREATE EXTENSION IF NOT EXISTS vector")

    async def _after_create_schema(self, conn: AsyncConnection) -> None:
        # document_id is the documents PK (already unique). The pgvector ANN index + the sparse FTS
        # GIN index (matching sparse_search's to_tsvector expression so the planner uses it).
        await conn.exec_driver_sql(
            "CREATE INDEX IF NOT EXISTS idx_embeddings_vector "
            f"ON embeddings USING ivfflat (vector vector_cosine_ops) WITH (lists = {self._IVFFLAT_LISTS})"
        )
        await conn.exec_driver_sql(
            "CREATE INDEX IF NOT EXISTS idx_chunks_fts ON chunks USING gin (to_tsvector('english', text))"
        )

    async def store_embeddings(self, embeddings) -> list[str]:
        """Upsert dense vectors into the ``embeddings`` table — ``ON CONFLICT (chunk_id)`` replaces,
        so re-embedding a chunk is idempotent (the Postgres counterpart of SQLite's
        ``INSERT OR REPLACE`` into ``vec_chunks``). Returns the chunk ids written."""
        if not embeddings:
            return []
        rows = [
            {"chunk_id": e.chunk_id, "vector": self._encode_vector(e.vector)} for e in embeddings
        ]
        stmt = pg_insert(self.embeddings).values(rows)
        stmt = stmt.on_conflict_do_update(
            index_elements=[self.embeddings.c.chunk_id], set_={"vector": stmt.excluded.vector}
        )
        async with self.engine.begin() as conn:
            await conn.execute(stmt)
        return [e.chunk_id for e in embeddings]

    async def dense_knn(
        self, query_vec: list[float], k: int, filter: ChunkFilter | None = None
    ) -> list[Candidate]:
        """
        §8 dense KNN over the pgvector ``embeddings`` table (cosine distance, nearest first), returned as
        ranked ``Candidate``s — the SQLite ``vec_chunks`` counterpart. The result is the ``k`` nearest
        *retrieval units* (leaf chunks) that pass ``filter``: a filtered ivfflat scan can under-return, so
        we over-fetch the top window, join ``chunks``, drop the rest, and backfill (``_overfetch``).
        """
        dist = self.embeddings.c.vector.cosine_distance(query_vec)
        cond = self._search_condition(filter)
        async with self.engine.begin() as conn:
            # ivfflat is approximate: it probes only `ivfflat.probes` lists (default 1), so an ORDER BY ...
            # LIMIT can return fewer rows than exist regardless of the LIMIT — over-fetch alone can't
            # backfill what the index never surfaces. Scan all lists so recall is exact and the over-fetch
            # sees the true nearest set (throughput is de-prioritized — NFR-2; SET LOCAL is scoped to this
            # transaction).
            await conn.exec_driver_sql(f"SET LOCAL ivfflat.probes = {self._IVFFLAT_LISTS}")
            total = (await conn.execute(select(func.count()).select_from(self.embeddings))).scalar_one()

            async def page(window: int) -> list[tuple[str, float]]:
                inner = (
                    select(self.embeddings.c.chunk_id, dist.label("distance"))
                    .order_by(dist)
                    .limit(window)
                    .subquery()
                )
                stmt = (
                    select(inner.c.chunk_id, inner.c.distance)
                    .join(self.chunks, self.chunks.c.chunk_id == inner.c.chunk_id)
                    .where(cond)
                    .order_by(inner.c.distance)
                )
                return (await conn.execute(stmt)).all()

            return await self._overfetch(k, total, page)

    async def sparse_search(
        self, query_text: str, k: int, filter: ChunkFilter | None = None
    ) -> list[Candidate]:
        """§8 sparse retrieval over a ``to_tsvector('english', text)`` GIN index (``ts_rank_cd``, best
        first). ``websearch_to_tsquery`` parses the raw text safely AND honors exact-match intent — a
        double-quoted span becomes a phrase query (adjacency-required, the Postgres counterpart of the
        SQLite builder's required phrases); unquoted terms are AND-ed exactly as ``plainto_tsquery`` did.
        Dotted identifiers (``6.4.2``) tokenize as single version/file lexemes on BOTH the index and the
        query side (the same ``english`` parser), so they match exactly without special handling.
        ``raw_score`` is ts_rank (higher better). Restricted to retrieval units (leaf chunks — the GIN
        index covers every chunk row, section parents included) and ``filter``, as in :meth:`dense_knn`
        (pre-filtered + over-fetched via ``_overfetch``)."""
        tsv = func.to_tsvector("english", self.chunks.c.text)
        tsq = func.websearch_to_tsquery("english", query_text)
        score = func.ts_rank_cd(tsv, tsq).label("score")
        base = select(self.chunks.c.chunk_id, score).where(tsv.op("@@")(tsq), self._search_condition(filter))
        async with self.engine.connect() as conn:
            total = (await conn.execute(select(func.count()).select_from(self.chunks))).scalar_one()

            async def page(window: int) -> list[tuple[str, float]]:
                return (await conn.execute(base.order_by(score.desc()).limit(window))).all()

            return await self._overfetch(k, total, page)

    def _search_condition(self, filter: ChunkFilter | None) -> ColumnElement[bool]:
        """The full search predicate over ``chunks``: retrieval units only (a section parent is hydrated,
        never searched) AND, when given, the permitted-chunk ``filter``."""
        leaf = self.chunks.c.level == 0
        return leaf if filter is None else and_(leaf, self._chunk_filter_condition(filter))

    def _chunk_filter_condition(self, filter: ChunkFilter) -> ColumnElement[bool]:
        """Build the permitted-chunk boolean over ``chunks`` (Core) for ``filter`` — the Postgres
        counterpart of ``SqliteRepository._chunk_filter_sql``. ``true()`` when nothing is restricted;
        an empty ``method_scope`` yields ``false()`` (nothing permitted)."""
        c = self.chunks.c
        conds: list[ColumnElement[bool]] = []
        if filter.require_available:
            conds.append(c.available == 1)
        if filter.require_grounding:
            conds.append(c.ai_grounding_allowed == 1)
        if filter.license_classes is not None:
            conds.append(c.license_class.in_(filter.license_classes) if filter.license_classes else false())
        if filter.method_scope is not None:
            mc = self.method_chunks.c
            if not filter.method_scope:
                conds.append(false())
            else:
                refs = or_(
                    *[
                        mc.method_id == ref.method_id
                        if ref.method_version is None
                        else and_(mc.method_id == ref.method_id, mc.method_version == ref.method_version)
                        for ref in filter.method_scope
                    ]
                )
                conds.append(c.chunk_id.in_(select(mc.chunk_id).where(refs)))
        return and_(*conds) if conds else true()

    async def hydrate(self, chunk_ids: list[str]) -> list[ChunkRecord]:
        """
        §8 hydration via SQLAlchemy over the normal tables (chunks joined to documents, plus
        method_chunks), preserving input order. No virtual tables here, so unlike SQLite this
        stays in Core.
        """
        if not chunk_ids:
            return []
        c, d = self.chunks, self.documents
        stmt = (
            select(
                c.c.chunk_id, c.c.text, c.c.document_id, d.c.source_kind,
                d.c.standard_id, c.c.locator, c.c.license_class,
                c.c.ai_grounding_allowed, c.c.available,
            )
            .select_from(c.join(d, c.c.document_id == d.c.document_id))
            .where(c.c.chunk_id.in_(chunk_ids))
        )
        async with self.engine.connect() as conn:
            by_id = {r[0]: r for r in (await conn.execute(stmt)).all()}
            prov = await self.reads.chunk_provenance(conn, chunk_ids)
            methods = await self.reads.methods_by_chunk(conn, chunk_ids)  # one query, not one per chunk
            records = [
                self.reads.create_chunk_record(r, methods.get(cid, []), prov.get(cid))
                for cid in chunk_ids
                if (r := by_id.get(cid)) is not None
            ]
        return records
